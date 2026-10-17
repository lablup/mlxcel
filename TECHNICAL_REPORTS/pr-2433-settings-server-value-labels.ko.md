# PR #2433: Settings > Server 값을 표시용 텍스트로 변환

**작성일**: 2026-10-17
**상태**: 머지 전에 CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료
**위험도**: 낮음 (WebUI 문구, 테스트 fixture, Rust 테스트 코드만 변경. 서빙 경로와 `/settings` 응답은 변경 없음)

## 요약

이슈 #2419는 #2334(PR #2344)의 후속이다. 해당 변경은 Settings > Server의 모든 읽기 전용 필드 라벨을 i18n 카탈로그로 바꿨지만, 라벨 옆의 값은 여전히 wire 표기였다. `expert_parallel`, `allow_list`, `true`와 `false`, JSON으로 출력되는 최상위 리스트가 그 예다. 이 PR은 같은 규칙을 값에도 적용한다. 영어와 한국어 모두에서 details 접기 영역 밖에는 내부 표기를 노출하지 않는다. enum 값은 카탈로그 라벨로, boolean은 예/아니요로, 단순 리스트는 쉼표로 구분한 한 줄로 표시한다.

## 문제 정의

`formatSettingValue`는 문자열이 아닌 모든 값을 `JSON.stringify`로 처리해서 `true`와 `[]`가 그대로 출력되었다. 최상위 행과 하위 행 헬퍼 `nestedValue`는 enum 문자열을 받은 그대로 출력했다. `nestedValue`는 이미 단순 리스트를 이어 붙이고 빈 리스트를 "설정 안 됨"으로 바꿨지만 최상위 경로에는 그 처리가 없었다. #2344 리포트가 이 공백을 알려진 한계로 적었고, 에픽 #1910 계약(`docs/webui/ux-contract.md`)은 내부 식별자를 사용자 문구로 보지 않는다.

## 기술적 선택과 그 이유

- enum 값마다 카탈로그 항목을 하나씩 둔다. `startup-strings.ts`의 `settings.startup.<path>.values.<spelling>`이며 17개 경로의 48개 표기를 en, ko로 제공한다. 카탈로그 키에 쓸 수 없는 문자는 `_`로 바꾼다. 예를 들어 `fp16+turbo4`는 `fp16_turbo4`로 조회한다.
- boolean은 `settings.value.yes`와 `settings.value.no`(Yes/No, 예/아니요)를 쓴다. 이슈가 제안한 `on`/`off` 키는 쓰지 않았다. "Batching off", "Values hidden", "Gemma 4 family model"처럼 이미 부정이거나 사실을 나타내는 라벨이 많아서 "Batching off: On"은 어색하게 읽힌다.
- 단순 리스트는 쉼표로 구분한 한 줄로 표시하고 빈 리스트는 "설정 안 됨"으로 표시한다. 쉼표 줄이 가릴 수 있는 데이터 항목(빈 문자열, 쉼표, 제어 문자, 가장자리 공백)은 따옴표로 감싸고, 중첩 리스트는 `(a, b)`로 표시한다.
- 표기별 라벨이 없는 경로의 문자열은 데이터(모델 이름, origin, 주소)이므로 받은 그대로 표시한다. 바이트 단위 10진수 KV cache 예산도 마찬가지다.
- 카탈로그에 라벨이 없는 표기는 기존 raw 값 접기 영역으로 옮긴다. 최상위든 부모 행 안이든 같다. 그래서 새 서버가 보낸 라벨 없는 표기가 메인 목록에 노출되는 일은 없다.
- 포매터는 새 파일 `webui/src/features/settings/value-text.ts`에 있다. 읽기 전용 행과 편집된 컨트롤의 "Server value" 힌트에 쓰인다. 입력창과 편집 여부 검사는 계속 `formatSettingValue`를 쓰므로 `parseSettingInput`을 통한 편집 왕복은 그대로다.
- `/props`의 "Active KV cache mode" 행은 `kv_cache_mode` 라벨을 재사용한다. 이 행에는 접기 영역이 없어서 알 수 없는 모드는 받은 그대로 표시한다.

트레이드오프: 라벨이 메인 번들에 포함된다. 초기 JS gzip은 200 KiB 예산 중 190.34 kB로 약 93%다(기존 91%). 지연 로딩하면 줄일 수 있지만, 첫 렌더링부터 문구가 있어야 하는 탭에 로딩 상태가 생긴다.

## 드리프트 가드

`tests/fixtures/settings/read-only-fields.json`에 `enum_values`(경로별 표기 목록)가 추가되었다. 새 Rust 테스트 `read_only_enum_fixture_lists_every_spelling`은 각 enum의 모든 variant를 포함하는 설정을 만든다. variant 목록은 exhaustive `match` 뒤에 있어서 새 variant를 목록에 넣기 전에는 컴파일되지 않고, 이후 테스트가 맵이 `read_only_value`의 출력과 같은지 검사한다. `startup-labels.test.ts`는 표기에 en 또는 ko 라벨이 없을 때, 라벨에 대응하는 표기가 없을 때, 한 경로의 두 표기가 같은 키로 합쳐질 때, enum 경로가 실제 필드가 아닐 때 실패한다. enum 값을 갖는 새 필드는 JSON에서 추론되지 않으므로 해당 테스트의 `ENUM_PATHS`에 경로를 직접 추가해야 한다. 후속 커밋에서 모든 enum 목록을 다루도록 테스트를 확장했다.

## 변경 내용

- `webui/src/features/settings/value-text.ts`(신규), `startup-strings.ts`, `strings.ts`: 포매터, 값 라벨, 병합.
- `webui/src/features/settings/server-settings.tsx`, `setting-control.tsx`: 읽기 전용 행과 server-value 힌트가 포매터를 사용하고, 알 수 없는 표기는 접기 영역으로 이동.
- `tests/fixtures/settings/read-only-fields.json`, `src/server/runtime_settings_values_tests.rs`: `enum_values`와 enum 드리프트 테스트.
- `startup-labels.test.ts`, `startup-values.test.tsx`, `setting-control.test.tsx`, `webui/tests/settings.spec.ts`: 라벨 커버리지, enum, 알 수 없는 표기, boolean, 리스트 케이스(en, ko)와 실서버 케이스.
- `tests/fixtures/webui/strings.json`, `docs/webui/settings.md`: fixture 동기화와 문서.
- `src/webui/assets`: `scripts/webui/build_bundle.py`로 재빌드.

## 동작 변경

WebUI 문구만 바뀐다. `/settings` 응답과 서빙 경로는 그대로다. enum 원본 표기, `true`/`false`, JSON 리스트는 이제 details 접기 영역 안에만 나타난다.

## 검증

- `pnpm lint`, `typecheck`, `unit`(744), `browser`(126), `verify-generated` 통과(Node 26.5.1, pnpm 11.18.0). Rust `server::runtime_settings`(14개 테스트, `--profile test-fast --features cuda`, `--test-threads=1`), `cargo fmt --check`, WebUI 계약 검사도 통과.
- 되돌리기 검사: `server-settings.tsx`와 `setting-control.tsx`를 `main`으로 되돌리면 새 vitest 케이스 9개가 모두 실패하고, 새 Playwright 케이스도 두 로케일에서 실패한다(라벨 대신 `hybrid`가 표시됨). `router` 라벨을 지우면 `startup-labels.test.ts`가 해당 키를 지목하며 실패하고, fixture에서 `router`를 지우면 Rust enum 테스트가 실패한다. 기존 `setting-control.test.tsx`의 왕복 케이스는 변경 없이 통과한다.
- 실서버: GB10에서 `mlxcel-server --webui --settings --props`와 `qwen3-0.6b-4bit`로 Playwright를 1440x900, 390x844, en, ko 조합에서 실행했다. 모든 경우에 라벨 행 120개, 접기 영역 밖의 enum 표기, boolean, JSON 리스트 없음, raw 값 접기 영역 없음, 가로 넘침 없음, `<html lang>` 일치를 확인했다. 기본값이 아닌 옵션(`--tp-moe-mode within_expert`, `--apc-hash blake3`, `--decode-storage-backend dense` 등)으로 다시 띄웠을 때도 "Within expert (each expert split across ranks)", ko "전문가 내부 분할(전문가마다 랭크에 나눔)" 같은 라벨이 표시되었다.

## 미검증 사항과 후속 조치

- `--cors-origins "http://localhost:5173,https://a.test"`는 쉼표가 들어간 origin 문자열 하나를 가진 `literal` 정책으로 보고된다. 리스트 포매터는 이를 따옴표로 감싸 표시한다. 서버 쪽 보고 방식의 문제이며 이 PR에서는 바꾸지 않았다.
- 번들 예산이 약 93%다. 다음 카탈로그 확장 때는 라벨 분리를 고려해야 한다.
- enum 값을 갖는 새 필드는 `startup-labels.test.ts`의 `ENUM_PATHS`에 직접 추가해야 한다.

## 변경 요약

| 항목 | 값 |
|------|-------|
| 변경 파일 | 소스, 테스트, 문서 13개와 재빌드된 asset 5개 |
| 라인 (asset 제외) | +990 / -46 |
| 커밋 | `86dd4dbf` feat(webui): show Settings > Server values as display text; `b0272c20` fix(webui): cover every enum list in the spelling drift test |
| 관련 | 이슈 #2419, PR #2344, 이슈 #2334, 에픽 #1910 |
