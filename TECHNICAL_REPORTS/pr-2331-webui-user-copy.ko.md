# PR #2331: WebUI 사용자 문구에 내부 상태가 출력되던 문제 수정

**날짜**: 2026-10-16
**상태**: 머지 전 CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료, 하위 행 라벨은 아직 원시 필드명 ("미검증 및 후속" 참고)
**위험도**: 낮음~중간 (opt-in `/settings` 응답 형태가 바뀌며, `/ui-api/v1`과 서빙 경로는 바뀌지 않음)

## 요약

이슈 #2329는 에픽 #1910의 Definition-of-done 점검에서 발견되었다. 에픽 계약은 내부 식별자, 원시 사유 문자열, enum 이름을 사용자 문구로 쓰는 것을 금지하는데, 세 곳이 이를 어기고 있었다.

1. Settings > Server 시작 값이 더 이상 `src/server/runtime_settings.rs`의 `format!("{value:?}")` Debug 클로저를 거치지 않는다. enum은 CLI 표기를 쓰고, 없는 값은 `null`이며, 구조화된 값은 라벨이 붙은 하위 행으로 렌더링되는 평탄한 JSON 객체다.
2. 사이드바 푸터가 서버 모드를 `serverModeLabel()`로 카탈로그 문자열에 매핑한다. 예: "Multi-model · ready · v0.7.0", "다중 모델 · 준비 · v0.7.0".
3. Stop 안내가 원시 타임스탬프 대신 `clockTime()`으로 `measured_at`을 포맷한다.

## 문제 정의

`read_only_value`가 시작 값 12개 필드를 `{value:?}`로 포맷해서, Server 탭에 enum variant 이름과 `Some(..)` 같은 Rust Debug 출력이 보였다. 푸터는 계약의 `ServerMode` 값을 그대로 출력했다. Stop 안내는 나노초 자릿수가 붙은 서버의 RFC 3339 타임스탬프를 그대로 넣었다. 셋 모두 구현 세부를 사용자에게 노출하며 현지화도 할 수 없다.

## 기술적 선택과 그 이유

- enum은 `src/server/config.rs`의 새 `as_str()` 메서드 또는 기존 Display로 CLI 표기를 쓴다(`longest-first`, `auto`, `any`, `hybrid`). 사용자가 읽는 값이 명령줄에 입력할 값과 같다.
- 없는 값은 JSON `null`이며 UI가 "not set"으로 표시한다. 서버가 "없음"을 뜻하는 문자열을 만들어내지 않는다.
- 구조화된 값(`tensor_parallel`, `prompt_cache`, `batch_kv_quant`, `cors_policy`, `gcp`, `pipeline_parallel_runtime`, `remote_pipeline_stage`)은 새 `src/server/runtime_settings_values.rs`에서 만드는 평탄한 JSON 객체이고, `kv_cache_budget`은 `--kv-cache-budget` 표기를 유지한다. 스키마 `type`은 이에 맞춰 `object`, `object_or_null`(또는 `str_or_null`)이 된다. 구조는 데이터이므로 렌더링은 UI가 정하고, JSON이나 Debug 문자열은 표시되지 않는다.
- Server 탭은 객체를 라벨이 붙은 하위 행으로 렌더링하고, 객체 안의 객체 목록은 항목별 하위 행으로 유지한다.
- `model_free`는 "Multi-model, no models"로 표시된다. 등록된 모델이 없을 때 라우터가 이 모드를 고르기 때문이다. 번들이 모르는 모드는 "Server" / "서버"로 표시되어, 더 새로운 서버가 오래된 번들에 원시 enum을 출력하지 않는다. 연결 요약 배너도 같은 라벨을 쓴다.
- `toDate()`는 파싱 전에 밀리초를 넘는 소수 자릿수를 잘라낸다. 서버가 나노초를 보내기 때문이다. 시간이 없거나 파싱할 수 없으면 기존처럼 `chat.announce.unknown_time`으로 대체한다.
- UI에서 Debug 문자열을 번역하지 않고 서버 출력을 고친다. Debug 텍스트를 파싱하면 UI가 Rust 포맷에 묶이기 때문이다.

## 변경 요약

- `src/server/runtime_settings.rs`: Debug 클로저 제거, 새 빌더의 값 사용.
- `src/server/runtime_settings_values.rs`(신규), `src/server/runtime_settings_values_tests.rs`(신규): 평탄한 객체 빌더와 values 테스트 4개.
- `src/server/config.rs`: CLI 표기가 없던 enum에 `as_str()` 추가.
- `webui/src/features/settings/server-settings.tsx`, `settings.css`: 하위 행 렌더링.
- `webui/src/provider-surfaces.tsx`, `webui/src/i18n/catalog.ts`: `serverModeLabel()`과 새 문자열.
- `webui/src/features/activity/format.ts`, `webui/src/features/chat/chat.tsx`: `toDate()` 절단과 Stop 안내의 `clockTime()`.
- 테스트와 fixture: `tests/fixtures/settings/startup-values.json`(신규), `tests/fixtures/webui/strings.json`, `startup-values.test.tsx`(신규), `provider-surfaces.test.ts`, `chat.test.tsx`, `app.provider.test.tsx`, `webui/tests/shell.spec.ts`.
- 문서: `docs/server-features.md`, `docs/webui/settings.md`, `docs/webui/design-system.md`. `verify-generated`를 위해 `src/webui/assets`의 번들을 다시 빌드했다.

## 동작 변경

opt-in `/settings` 응답 형태만 바뀌며 `docs/server-features.md`에 문서화했다. 위 12개 필드는 Debug 텍스트 대신 CLI 표기 문자열, `null`, 평탄한 객체를 반환한다. 이전 Debug 문자열을 파싱하던 클라이언트는 새 형태로 옮겨야 한다. `/ui-api/v1`은 바뀌지 않았다.

## 검증

실제 체크포인트 `qwen3-0.6b-4bit`, 번들된 `test-fast` `mlxcel-server`, Playwright로 1440x900(en, ko)와 390x844에서 확인했다.

- Server 탭은 로케일당 값 셀 120개이며 Debug 형태는 없다. 모바일 레이아웃에 가로 스크롤이 없다.
- 푸터는 "Multi-model · ready · v0.7.0"과 "다중 모델 · 준비 · v0.7.0"으로 표시된다. Stop 안내는 두 로케일 모두 포맷된 시각을 보여준다.
- 되돌리기 검사: Debug 클로저를 되살리면 Rust values 테스트 4개가 실패한다. 이전 소스로 되돌리면 startup-values, provider-surfaces, chat Stop 시각 웹 테스트가 실패한다.
- 게이트: `pnpm` lint, typecheck, unit(722), browser(124), verify-generated 통과. clippy, fmt, 계약 검사 통과.

## 미검증 및 후속

- 하위 행 라벨은 `tp_size` 같은 원시 필드명이며, 기존 최상위 읽기 전용 라벨과 같다. 읽기 전용 설정의 라벨 현지화는 별도 UX 변경이다.
