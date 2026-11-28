# 기술 보고서: PR #2904 - fix(server): pin Claude Code billing stamp in Anthropic system text

**작성일**: 2026-11-09
**리뷰어**: -
**상태**: 완료
**언어**: Rust
**위험도**: 낮음

---

## 요약

Claude Code는 시스템 프롬프트 맨 앞에 billing 헤더를 넣는다. 이 헤더의 `cch` 값은 요청마다, `cc_version`의 네 번째 부분은 세션마다 바뀐다. 이제 `mlxcel-server`는 프롬프트를 렌더링하기 전에 이 두 값을 같은 길이의 `f` 문자로 덮어써서, 연속된 Anthropic Messages 요청이 prefix를 공유하게 한다. 실제 체크포인트 두 개에서 2번째 턴의 재사용 토큰은 0에서 3789/3821(`qwen3.8-27b-4bit`, 스냅샷 재사용)과 1984/2007(`gemma-3n-e2b-it-4bit`, KV 재사용)로 늘었다.

---

## 1. 문제 정의

### 1.1 배경

프롬프트 캐시는 렌더링된 토큰을 키로 쓰므로 한 바이트만 바뀌어도 그 위치에서 재사용이 끊긴다. 스탬프 형식은 다음과 같다.

```
x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=sdk-cli; cch=b145e;
```

### 1.2 기존 문제점

- `AnthropicSystem::to_text` 결과가 그대로 선행 시스템 메시지가 됐고, `src/` 어디에서도 스탬프를 처리하지 않았다.
- 시스템 텍스트를 먼저 렌더링하는 템플릿에서는 2번째 턴이 캐시를 완전히 놓쳤다. 도구를 먼저 렌더링하는 템플릿에서는 도구 목록 뒤에서 갈라진다(Strata 측정 기준 약 22K 토큰 지점).

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| Claude Code의 매 턴이 시스템 프롬프트와 히스토리를 다시 읽음 | 높음 | Claude Code 클라이언트에서 높음 |
| 모델이 바뀐 스탬프 텍스트를 봄 | 낮음 | 설계상 항상 발생 |

---

## 3. 기술적 선택과 그 이유

### 3.1 변하는 두 값만 같은 길이의 `f`로 고정

**컨텍스트:** llama.cpp(ggml-org/llama.cpp#21793)는 `cch`만 고정하고, Strata는 두 값을 모두 고정한다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 헤더 제거 | 프롬프트가 가장 짧음 | 모델이 보는 구조가 바뀜 |
| `cch`만 고정 | 수정 범위가 가장 작음 | 세션이 바뀌면 재사용이 끊김 |
| **선택: `cch`와 `cc_version` 세션 부분 고정** | 세션 안팎에서 안정적, 토큰 형태 유지 | 한 클라이언트 형식 전용 |

**선택 이유:** 대소문자를 구분하는 정확한 prefix가 있고, 첫 160바이트의 첫 줄 안에서, 길이 제한을 지키며 `;`로 끝나는 ASCII 값에만 동작한다. 그 밖의 입력은 그대로 통과하므로 일반 시스템 프롬프트에는 영향이 없다.

**트레이드오프:** 일부러 범용 정규화기로 만들지 않았다. 프롬프트에 스탬프를 찍는 다른 클라이언트가 나오면 별도의 문서화된 규칙이 필요하다. 이슈는 문자 수로 적었지만 구현은 바이트 수로 세며, 바이트가 더 엄격한 제한이다.

### 3.2 translator에서 적용, 기본 활성화, kill switch 제공

`anthropic_request_to_chat`에서 평탄화된 텍스트에 적용하므로 `system`의 문자열 형식과 블록 배열 형식을 모두 처리한다. `messages` 안의 `role: "system"` 턴은 고정하지 않는다. `MLXCEL_ANTHROPIC_PIN_BILLING_STAMP=0`이면 원문을 그대로 전달하며, 프로세스당 한 번만 읽는다.

---

## 4. 구현 상세

| 파일 | 변경 |
|------|------|
| `src/server/anthropic_billing_stamp.rs` | 신규: `Cow<str>`를 반환하는 `pin_billing_stamp`, 필드 파싱, kill switch, 단위 테스트 |
| `src/server/anthropic_translator.rs` | 최상위 `system` 텍스트 고정, translator 테스트 |
| `docs/server-features.md`, `docs/environment-variables.md` | 동작과 스위치 문서화 |

ASCII 값의 바이트 단위로 매칭하므로 헤더 뒤의 비ASCII 텍스트는 건드리지 않고, 치환 범위도 문자 경계에 놓인다. 필드 키는 헤더의 콜론, 공백, `;` 바로 뒤에 있어야 한다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 5 |
| 추가 줄 | +383 |
| 삭제 줄 | -2 |
| 추가 테스트 | 신규 모듈 18개, translator 5개 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `2e36a6df` | fix | fix(server): pin Claude Code billing stamp in Anthropic system text |

---

## 8. 후속 조치

### 모니터링 필요

- 도구를 먼저 렌더링하는 템플릿으로 실제 `claude` CLI를 `mlxcel-server`에 연결해 도구 목록 너머까지 재사용되는지 확인한다. 기록된 실행은 CLI가 아니라 Claude Code 형태로 만든 요청을 사용했다.

### 향후 개선 사항

- 공백이나 `=`가 들어간 악의적 값은 두 편집 범위를 겹치게 만들 수 있어, 그런 입력에서는 고정 동작이 멱등이 아니다(적대적 코퍼스의 약 1.8%). panic은 불가능하고 보낸 사람 자신의 프롬프트에만 영향을 준다. 값에서 해당 문자를 거부하면 없앨 수 있다.
- 스탬프가 있는 요청은 시스템 텍스트를 한 번 더 복사한다(16 MiB 기준 약 280 µs).

---

## 부록

### A. 테스트 결과

- `cargo test --lib --profile test-fast --features metal,accelerate anthropic`: 111개 통과.
- 최종 커밋에서 `cargo test --lib server::anthropic`: 57개 통과.
- `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`: 경고 없음.
- `cargo fmt --check`, `make verify-env-var-docs`: 통과.
- PR 단위로 실행하지 않은 항목: 이슈의 워크스페이스 전체 테스트와 clippy 명령. 머지 후 오케스트레이터의 최종 게이트에서 다룬다.

### B. 측정

스탬프가 다른 Claude Code 형태의 `/v1/messages` 요청 두 개(블록 배열 `system`, 도구 12개, 릴리스 바이너리)의 2번째 턴이다. kill switch 쪽은 변경 전 요청 경로와 같으므로 "변경 전"으로 사용했다.

| 모델 | 변경 전(`=0`) | 변경 후(기본값) |
|------|---------------|-----------------|
| `qwen3.8-27b-4bit` | 0/3825 | 3789/3821 |
| `gemma-3n-e2b-it-4bit` | 0/2012 | 1984/2007 |

`gemma-3n-e2b-it-4bit`의 재사용은 첫 프롬프트 1984토큰 전체를 덮는다. `qwen3.8-27b-4bit`는 첫 프롬프트 3794토큰보다 5토큰 앞에서 멈추며, 그 원인은 독립적으로 확인하지 못했다. 이 템플릿들은 스탬프를 앞부분에 렌더링하므로 "변경 전"은 부분 재사용이 아니라 완전한 미스다.

### C. 참고 자료

- 이슈 #2884
- ggml-org/llama.cpp#21793
- Strata `serve/frontend.py`, `pin_billing_stamp`, `146df3c` 기준
