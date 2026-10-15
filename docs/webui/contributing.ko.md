# WebUI에 기여하기

[English](contributing.md) | [WebUI 문서 목록](README.md)

번들 WebUI를 바꿀 때 참고하는 안내 문서입니다. 기준은 링크된 문서이며, 이 문서는 그 계약을 다시 적지 않습니다.

## 파일 위치

| 경로 | 내용 | 직접 수정 |
|---|---|---|
| `webui/` | WebUI의 React·TypeScript 소스, 테스트, 버전이 고정된 `package.json` / `pnpm-lock.yaml` | 예 |
| `src/webui/assets/` | Cargo가 바이너리에 임베드하는 생성된 배포 번들. `mlxcel-webui-manifest.json`, `third-party-licenses.txt` 포함 | 아니요, 다시 생성 |
| `src/server/webui/` | Rust 쪽 정적 라우터, 보안 래퍼, typed UI API 어댑터 | 예 |
| `docs/webui/api.yaml` | `{api_prefix}/ui-api/v1`의 기계 판독용 계약 | 예, 계약부터 |
| `docs/webui/generated/ui-api.d.ts` | `api.yaml`에서 생성한 TypeScript 선언 | 아니요, 다시 생성 |
| `tests/fixtures/webui/` | Rust와 프런트엔드 테스트가 함께 쓰는 실행 가능한 계약 예시와 영어/한국어 문자열 카탈로그 픽스처 | 예 |
| `webpage/` | 홍보용 웹사이트(Next.js). WebUI가 아니며 어떤 바이너리에도 들어가지 않고 `webui/`와 코드를 공유하지 않습니다. | 별도 프로젝트 |

## 계약부터 바꾸기

API 변경은 생산자 스키마(`docs/webui/api.yaml`), 생성된 선언, 픽스처, 최소 한 곳의 소비자를 한 PR에서 함께 고칩니다. `python3 scripts/ci/check_webui_contract.py`가 드리프트 검사이고(`--fix`는 선언만 다시 생성), `make verify-webui-contract`는 격리된 환경에서 이를 실행합니다. 소유권과 변경 규칙은 [architecture.md](architecture.md)에, 생명주기와 재생 규칙은 [state-machines.md](state-machines.md)에 있습니다.

## 공유 모듈 사용

- `webui/src/api`가 유일한 전송 계층입니다. 화면은 백엔드 URL이나 자체 fetch 래퍼를 만들지 않습니다.
- `webui/src/state`가 유일한 상태 기준입니다(`WebUiProvider`, `useWebUi()`, `useWebUiActions()`). 화면은 자체 폴링 타이머, 저장, 모델 캐시를 추가하지 않습니다.
- `webui/src/design-system`이 `@lablup/ui-common`을 감쌉니다. 기능 화면은 패키지를 직접 가져오지 않고 이 계층의 기본 요소를 씁니다. [ui-common.md](ui-common.md)와 [design-system.md](design-system.md)를 참고하세요.
- 사용자에게 보이는 모든 문자열은 `webui/src/i18n/catalog.ts`에 영어와 한국어 값을 가진 키로 있어야 하고, `tests/fixtures/webui/strings.json`과 일치해야 합니다([ux-contract.md](ux-contract.md)).

화면별 동작은 [catalog.md](catalog.ko.md)(모델), [chat.md](chat.md), [activity.md](activity.md), [settings.md](settings.md), [runtime.md](runtime.md)에 있습니다.

## 프런트엔드 빌드와 자산 재생성

도구 버전은 정확히 고정되어 있습니다. Node 26.5.1, pnpm 11.18.0입니다. 저장소 루트에서 실행합니다.

```bash
pnpm --dir webui install --frozen-lockfile
pnpm --dir webui run typecheck
pnpm --dir webui run lint
pnpm --dir webui run unit
python3 scripts/webui/build_bundle.py
make verify-webui-bundle
```

`build_bundle.py`는 도구 버전을 확인하고 빌드한 뒤 매니페스트를 쓰고 예산을 검증합니다. 다시 생성한 `src/webui/assets/`는 소스 변경과 함께 커밋합니다. `make verify-webui-bundle`은 깨끗한 상태에서 두 번 다시 빌드해 커밋된 자산과 바이트 하나라도 다르거나, 소스 다이제스트가 오래됐거나, 예산을 넘으면 실패합니다. `pnpm --dir webui run build`만으로는 배포용 빌드가 아닙니다. 자세한 내용은 [bundling.ko.md](bundling.ko.md)에 있습니다.

## Rust 빌드는 오프라인

Cargo는 `src/webui/assets/`에 커밋된 파일을 임베드할 뿐 Node를 실행하거나 프런트엔드 패키지를 내려받지 않습니다. 따라서 Rust만 다루는 기여자는 JavaScript 도구가 필요 없습니다. UI는 기본으로 켜진 `webui` Cargo feature 뒤에 있습니다.

| 빌드 | WebUI |
|---|---|
| `cargo build --release --features metal,accelerate` | 포함(기본 feature) |
| `cargo build --release --features cuda` | 포함(기본 feature) |
| `cargo build --release --no-default-features --features metal,accelerate,surgery` | 제외. 이때 `--webui`는 "requires a binary built with the Cargo `webui` feature" 메시지와 함께 시작에 실패합니다. |
| `cargo build --release --no-default-features --features metal,accelerate,webui` | 다른 기본 feature 없이 포함 |

## 검증

`make verify-webui`는 CPU만으로 도는 종합 검사입니다. 계약 드리프트, 검증 도우미 테스트, 프런트엔드 타입·린트·단위·브라우저 검사, 번들 재현성, 범위를 좁힌 Rust WebUI 테스트를 실행합니다. 설치 산출물, 리버스 프록시, 시작 시간, 실제 모델, CUDA 근거는 [통합 매트릭스](../webui-integration-matrix.md)에 나열된 별도 대상이며, 릴리스에 무엇이 필요한지도 그 문서에 있습니다. Rust 변경은 저장소 전체의 `test-fast`, clippy, fmt 검사도 통과해야 합니다.

## 라이선스

번들에 들어가는 프런트엔드 런타임 의존성은 `webui/public/third-party-licenses.txt`(번들에 복사됨)에 라이선스 전문을, 최상위 `NOTICE`에 항목을 두어야 합니다. 다른 프로젝트에서 가져와 고친 파일은 `webui/src/design-system/themes/glass.css`처럼 출처를 밝히는 머리 주석을 둡니다. 글꼴 파일, Apple 디자인 키트 자산, SF Symbols, 직접 만들지 않은 아이콘은 추가하지 마세요. WebUI는 시스템 글꼴과 프로젝트에서 만든 SVG 아이콘을 씁니다.
