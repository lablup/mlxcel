# PR #2344: Server 설정 하위 행 라벨 현지화

**작성일**: 2026-10-16
**상태**: 머지 전에 CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. 라벨이 아닌 값은 여전히 wire 표기를 보여 줌 ("미검증 및 후속 조치" 참고)
**위험도**: 낮음 (WebUI 문구, 테스트 fixture, Rust 테스트 코드만 변경. 서빙 경로와 `/settings` 응답은 그대로)

## 요약

이슈 #2334는 #2329(PR #2331)의 후속이다. Settings > Server는 읽기 전용 시작 값과 #2329에서 추가한 구조화 하위 행을 `tensor_parallel`, `tp_size` 같은 원시 필드 이름으로 표시했고, 관측된 컨텍스트 행은 `/props n_ctx`, `/props total_slots`를 그대로 출력했다. 에픽 #1910 계약(`docs/webui/ux-contract.md`)은 내부 식별자를 사용자 문구로 취급하지 않는다. 이 PR은 모든 읽기 전용 필드와 하위 필드를 i18n 카탈로그의 영어와 한국어 라벨로 표시하고, 원시 이름은 details 공개 영역 안에만 둔다.

## 문제 정의

읽기 전용 시작 필드 90개와 구조화 값의 하위 필드 45개에 카탈로그 키가 없어서 탭이 서버의 snake_case 이름을 그대로 라벨로 출력했다. 서버에 필드가 추가되어도 실패하는 테스트가 없어서 누락은 늘어나기만 했다.

## 기술적 선택과 그 이유

- 필드마다(`settings.startup.<field>`), 하위 필드마다(`settings.startup.<field>.<sub_field>`) 카탈로그 항목을 하나씩 둔다. 새 파일 `webui/src/features/settings/startup-strings.ts`에 정의하고 `strings.ts`를 통해 `settingsStrings`에 병합하며 `tests/fixtures/webui/strings.json`에 반영한다. 키가 필드 경로에서 바로 만들어지므로 별도 매핑 표가 필요 없다.
- `setting-control.tsx`의 `startupLabel(locale, path)`는 라벨이 없거나 경로 조각에 키 구분자가 들어 있으면 `null`을 반환한다. 점이 들어간 이름이 다른 필드의 키로 해석되면 안 되기 때문이다.
- 라벨이 없는 필드나 하위 필드는 버리지 않는다. 원시 이름 그대로 "Other values (internal names)" details 영역으로 옮긴다. 최상위에서는 목록 아래에, 하위 필드는 부모 행 안에 둔다. 따라서 새 서버가 보낸 값이 이전 번들에서 사라지지 않고, 원시 이름이 주 라벨에 나오지도 않는다.
- 구조화 항목의 최상위 목록은 JSON 대신 라벨이 붙은 하위 행으로 표시한다.
- "Why these values are read-only" 영역은 각 필드를 `Label (raw_name)` 형태로 나열해 원시 이름이 필요한 사람도 볼 수 있다.
- 주 목록의 라벨이 이제 문장형이므로 monospace 키 스타일을 제거했다.
- 컨텍스트 라벨("Per-slot context tokens", "Configured slots", "Active KV cache mode")에서 `/props` 이름을 뺐다. "Reported context geometry" 영역 안의 새 `settings.server.context.source` 줄이 그 이름을 알려 준다.
- Rust fixture `tests/fixtures/settings/read-only-fields.json`이 모든 읽기 전용 `/settings` 필드와 그 값이 가질 수 있는 하위 필드를 나열한다. Rust 테스트는 fixture가 스키마와 같은지, 웹 테스트는 fixture가 카탈로그와 같은지 확인한다. 새 서버 필드는 fixture에 추가하기 전까지 Rust CI에서, 라벨을 추가하기 전까지 웹 테스트에서 실패한다.
- 리뷰 중 수정: `model_alias`와 `model_aliases`는 `src/server/config.rs` 기준으로 서빙 모델 이름과 전체 `--alias` 목록이다. 파이프라인 단계는 한국어로 "스테이지"로 표기하고, 스케줄러 필드 두 개의 한국어를 다듬었으며, fallback 영역 이름을 바꿨다.

트레이드오프: 두 로케일의 라벨 135개가 메인 번들에 포함된다. 초기 JS gzip이 약 4.7 KB 늘어 200 KiB 예산의 약 91%가 된다. 라벨을 지연 로딩하면 줄일 수 있지만, 첫 렌더링에 라벨이 있어야 하는 탭에 로딩 상태가 생긴다.

## 변경 내용

- `webui/src/features/settings/startup-strings.ts`(신규), `strings.ts`: 카탈로그 항목과 병합.
- `webui/src/features/settings/setting-control.tsx`: `startupLabel`.
- `webui/src/features/settings/server-settings.tsx`, `settings.css`: 라벨 조회, fallback 영역, 레코드 목록 하위 행, `Label (raw_name)` 사유, 컨텍스트 출처 줄, 하위 행 전체 폭의 하위 필드 영역.
- `tests/fixtures/settings/read-only-fields.json`(신규), `src/server/runtime_settings_values_tests.rs`: 기본, 설정됨, remote-coordinator 서버 구성(`pipeline_parallel_runtime` 두 형태 포함)에 대한 `read_only_field_fixture_lists_every_field_and_sub_field`.
- `webui/src/features/settings/startup-labels.test.ts`(신규): 키 누락, 영어 또는 한국어 값이 빈 경우, snake_case 라벨, 서버 필드가 없는 `settings.startup.*` 키에서 실패한다.
- `startup-values.test.tsx`, `webui/tests/settings.spec.ts`: 현지화 라벨 검증, 탭 전체에서 이름, 제목, 라벨, 요약이 snake_case가 아님을 확인, 알 수 없는 하위 필드와 목록 사례.
- `tests/fixtures/webui/strings.json`, `docs/webui/settings.md`: fixture 동기화, 라벨과 공개 영역, fixture와 테스트 쌍에 대한 문서.
- `src/webui/assets`: `scripts/webui/build_bundle.py`로 재생성(Node 26.5.1, pnpm 11.18.0).

## 동작 변경

WebUI 문구만 바뀐다. `/settings` 응답과 서빙 경로는 그대로다. 원시 필드 이름은 이제 details 영역 안에만 나타난다.

## 검증

- `pnpm lint`, `typecheck`, `unit`(732), `browser`(124), `verify-generated` 통과. Rust `server::runtime_settings` 테스트 13개(`--profile test-fast --features cuda`, `--test-threads=1`)와 `cargo fmt --all -- --check` 통과.
- 되돌리기 검증: `server-settings.tsx`를 `main`으로 되돌리면 당시 `startup-values.test.tsx`의 9개 중 7개가 실패한다. `tensor_parallel.tp_size` 항목을 지우고 한국어 값 하나를 비우면 `startup-labels.test.ts`가 실패하며 정확히 그 키를 보고한다. 이전 `/props n_ctx` 라벨을 복원하면 탭 전체 테스트가 두 로케일 모두에서 실패한다.
- 실제 서버: GB10에서 `test-fast` `mlxcel-server`에 `qwen3-0.6b-4bit`를 올리고 Playwright로 1440x900과 390x844, en과 ko에서 확인했다. 모든 경우에 카탈로그 라벨 120개, details 밖의 snake_case 텍스트 없음(요소 133개 검사), fallback 영역 없음, 가로 넘침 없음, `<html lang>` 일치를 확인했다.

## 미검증 및 후속 조치

- 값은 여전히 wire 표기를 보여 준다. `expert_parallel`, `allow_list` 같은 enum 값, 번역되지 않은 `true`와 `false`, `formatSettingValue`가 JSON으로 출력하는 최상위 단순 목록이 해당한다. 같은 계약이 적용되며 별도 이슈가 필요하다.
- 번들 예산이 약 91%에 도달했다. 다음 카탈로그 확장 때는 라벨 분리를 검토해야 한다.
