# 기술 보고서: PR #1864 — 공유 WebUI 클라이언트와 상태 관리 경계

**날짜**: 2026-09-12
**갱신**: 2026-09-13 — 웨이브 2 통합 검증
**상태**: 머지 전 구현 리뷰 완료, 후속 애플리케이션 통합은 남아 있음
**언어**: TypeScript, TSX, Rust, Python, JSON 호환 YAML
**위험 수준**: 중간
**관련 항목**: [PR #1864](https://github.com/lablup/mlxcel/pull/1864), [이슈 #1842](https://github.com/lablup/mlxcel/issues/1842), [에픽 #1834](https://github.com/lablup/mlxcel/issues/1834)

## 요약

번들 WebUI가 공통으로 사용할 브라우저 전송 계층과 헤드리스 React 상태 관리 경계를 구현했다. 정식 계약에 따른 런타임 검증, 인증 헤더를 사용하는 fetch 기반 SSE, 스냅샷 재동기화, 페이지용 훅을 제공하며 기존 Rust 코디네이터에 서버 인스턴스와 숫자 시퀀스를 묶은 재생 커서를 추가했다. 운영 페이지 마운트나 별도 모델 레지스트리는 구현하지 않았으며 브라우저·하드웨어 수용 검증 완료를 의미하지 않는다.

## 1. 문제 정의

Models, Chat, Activity, Settings는 인증, 모델 식별자, 수명주기, 진행 중 작업에 대해 같은 상태를 보아야 한다. 페이지별 fetch 래퍼와 타이머는 자격 증명을 중복 보유하고 관측 요청을 겹치게 하며 재연결·서버 재시작 이후 서로 다른 상태를 만들 수 있다. 특히 모델 로드나 다운로드 POST 직후 연결이 끊기면 서버 수락 여부를 모른 채 재전송하여 사용자 동작을 두 번 실행할 위험이 있다.

각 스냅샷은 동시에 만들어지지 않는다. 가장 최근의 전역 이벤트부터 재생하면 작업 스냅샷 이후, 더 새로운 카탈로그 스냅샷 이전에 발생한 작업 전이를 놓칠 수 있다. 불투명 이벤트 ID는 숫자로 비교할 수 없고 커서는 반드시 서버 인스턴스와 함께 해석해야 한다.

## 2. 기술적 선택과 이유

### 정식 스키마를 사용하는 경계 검증

`webui/src/api/jsonSchema.ts`는 정식 계약에 필요한 스키마 구문을 평가하고 `validation.ts`는 별도의 수작업 DTO 형태 대신 타입이 있는 검증 함수를 제공한다. 프런트엔드 테스트는 정상 응답의 일부 속성만 보는 것이 아니라 시나리오, 식별자, 문자열 카탈로그를 포함한 공유 fixture 41개 전체를 읽는다. 음성 테스트는 추가 속성, 필수 nullable 필드 누락, 잘못된 타임스탬프, 잘못된 판별자를 검사한다. 스키마와 생성 타입 선언도 결정적 번들의 소스 해시에 포함했다.

통합된 카탈로그는 원본 `metadata.model_type`, 해석된 `metadata.architecture`, 제한된 `metadata.declared_architectures`를 별개의 사실로 유지한다. 소비자 회귀 테스트가 이 값들과 필수 metadata·unknown-reason·removal 정보, 선언 개수·문자열 길이 제한을 검사한다. 보안·카탈로그 변경과의 통합은 기존 테스트 fixture를 갱신하고 중복 Rust helper·테스트를 제거하며 서로 다른 복사본을 유지하지 않는다.

새 런타임 검증 의존성을 피하는 대신, 평가기는 이 저장소의 정식 스키마에 필요한 범위로 제한된다. 범용 JSON Schema 구현이 아니므로 새 스키마 구문을 도입할 때 평가기와 독립 음성 테스트도 함께 확장해야 한다.

### 단일 인증 전송 계층

`WebUiApiClient`는 검증한 동일 출처 경로 접두사만 받고 bearer 인증을 헤더로 전송하며 리다이렉트와 외부 API 주소를 거부한다. 자격 증명은 메모리에만 보관한다. 401은 인증을 지우고 진행 중 작업을 중단하며 403은 권한 거부로 유지한다. 크기 제한 JSON 읽기와 점진적 SSE 파싱으로 버퍼 사용을 제한한다. fetch를 사용하므로 EventSource URL에 키를 넣지 않고 Authorization을 전달할 수 있다.

동일 전송 계층이 `/v1/chat/completions`와 `/v1/responses`도 지원한다. 추론 훅은 불투명 카탈로그 ID를 현재 요청용 `inference_id`로 해석하고 streaming과 `autoload=false`를 강제하며 content, reasoning, tool, usage 프레임을 후속 Chat 표시 계층에 그대로 전달한다. 렌더링, 대화 영속화, 도구 실행은 추가하지 않았다.

### 리소스별 시퀀스 경계와 원자적 재생 구독

리듀서는 카탈로그 스냅샷, 작업 스냅샷·레코드, 모델 revision, 런타임 스냅샷의 경계를 독립적으로 관리한다. 동기화기는 관측한 최대 이벤트가 아닌 보유 경계의 최솟값과 `server_instance_id`로 재개한다. 페이지로 나누어 읽는 카탈로그·작업 목록은 서버 인스턴스와 스냅샷 시퀀스가 모두 일치해야 일관된 목록으로 적용한다.

Rust 재생 검증, 코디네이터 구독, 스키마, 생성 TypeScript, 요청 fixture를 함께 변경했다. 쌍을 이룬 쿼리 인자가 주 커서이며 불투명 `Last-Event-ID`는 별도의 기존 방식으로만 남긴다. 누락, 중복, 충돌, 안전한 정수 범위 초과, 미래 커서는 타입이 있는 오류를 반환하고 다른 서버 인스턴스나 보존 링의 공백은 재스냅샷 신호를 반환한다. 이력 읽기와 실시간 구독 사이에서 이벤트가 사라지지 않도록 구독과 재생을 조정했다.

### 페이지별 저장소 대신 공유 헤드리스 훅

`WebUiProvider`, `useWebUi`, `useWebUiActions`가 인증 세션, 선택 모델, 스냅샷, 작업 재조정, 스트림 정리를 소유한다. 모델 동작의 Promise 완료는 준비 완료가 아니라 수락을 의미하므로 페이지는 공유 상태에서 최종 결과를 표시해야 한다. 결과를 모르는 POST는 다시 보내지 않고 원래 멱등성 키로 작업 목록과 대조하며 60초 재조정 기간 이후에도 해결되지 않으면 명시적으로 드러낸다.

겹치지 않는 단일 새로고침 루프는 보이는 탭에서 2초, 숨겨진 탭에서 30초 간격을 사용한다. 재연결 지연에는 상한과 지터가 있고 세대 검사는 정리·재시작 이전의 낡은 작업을 무시한다. 모델 선택과 로그아웃은 진행 중 전송을 중단한다. `lastUpdatedAt`은 상태 전이 시각이고 별도의 nullable `lastSuccessfulAt`은 검증을 통과한 현재 세션의 데이터 스냅샷·이벤트를 수용했을 때만 갱신된다. heartbeat, 무효화만 알리는 이벤트, 오류는 낡은 데이터를 최신으로 보이게 하지 않는다. 같은 인스턴스의 공백·오류는 이전 값을 유지하고 로그아웃·서버 교체는 값을 지운다. 아키텍처 문서에 두 시각의 의미와 전체 공개 훅을 명시했다.

## 3. 리뷰와 검증

이슈 구현 워크플로에서 세 차례 수정 후 독립 정확성·보안 리뷰가 잔여 지적 없이 완료되었다. 루트는 전체 workspace 검증을 위해 Rust revision `c735bc99`를 고정했다. 문서 최종 정리에서 마지막 데이터 수신 성공 시각의 수용 기준 공백을 발견했고 후속 `cfabec5c`와 `e728c9c9`가 TypeScript에 별도 시각과 회귀 테스트를 추가하고 실제 데이터에 적용하지 않은 알림을 최신성 갱신에서 제외했다. 최종 최신성 변경도 독립 리뷰 승인을 받았다. Rust 소스는 바꾸지 않았으며 최종 정리 담당자는 문서만 변경했다. 웨이브 2 통합에서 보안·카탈로그 변경을 포함한 `5505aae6` 위로 리베이스하고 `7497da98`를 게시했다. 통합 이전 `c735bc99`의 과거 게이트(11,167개 통과, 실패 0개, 무시 361개, workspace Clippy 통과)는 아래 결합된 검증을 대신하지 않는다. 두 독립 통합 리뷰는 `7497da98`를 잔여 지적 없이 승인했다.

| 검사 | 결과와 범위 |
|---|---|
| `pnpm --dir webui run typecheck` | 엄격한 TypeScript 검사 통과. |
| `pnpm --dir webui run lint` | ESLint 경고 없이 통과. |
| `pnpm --dir webui run unit` | 7개 파일의 45개 테스트 통과. 전체 fixture, SSE 바이트 분할·CJK·CRLF, 인증·중단, 리듀서, 페이지 읽기, 재조정, provider 정리, 데이터 수신 성공 시각을 포함한다. |
| `pnpm --dir webui run verify-generated` | 결합 게이트에서 깨끗한 임시 빌드 두 번과 커밋된 번들의 일치 확인. 검증 해시 `e0a37519f1c21dc11ca6b6b0b163fcb860e860d94b747d027dc1fe22c5ae8b11`. 문서 최종 정리에서는 자산을 다시 쓰지 않았다. |
| `pnpm --dir webui run browser` | 루트가 Vite preview 대상 Chromium shell 테스트 1개 통과를 확인했다. 실제 Safari나 운영 제어 API 통합 검증은 아니다. |
| `cargo check --workspace --no-default-features --features metal,accelerate` | 루트가 경고 43개와 함께 성공을 보고했다. 이 설정은 경고 없는 빌드가 아니다. |
| `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` | fixture 41개, 생성 DTO 드리프트, 엄격성·음성 검사 통과. |
| `make verify-llama-compat verify-versions verify-kernel-dtype-keys` | 격리 Python 환경을 PATH 앞에 두고 모두 통과. |
| `cargo fmt --check` | 통과. |
| `make verify-test` | 루트가 `7497da98`에서 실행한 CI와 동일한 결합 게이트: 123개 요약 기준 11,226개 통과, 실패 0개, 무시 361개. 무시된 테스트는 통과가 아니다. |
| `cargo test --profile test-fast --features webui ui_events` | 결합 게이트 전 검증에서 선택된 Rust 라우트 테스트 3개 통과. |
| `cargo test --profile test-fast --features webui lifecycle_coordinator_sequence_replay` | 결합 게이트 전 검증에서 선택된 Rust 코디네이터 테스트 2개 통과. |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | 구현 워크플로가 최종 수정 후 범위 제한 lint 통과를 보고했다. |
| `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings` | 루트가 `7497da98`의 결합 workspace·all-target 게이트 통과를 보고했다. |

고정한 런타임 `7497da98`의 CI와 동일한 결합 검증 체인이 종료 코드 0으로 완료되었다. workspace 테스트, all-target Clippy, 엄격한 fixture·정적 게이트, 기능 비활성 검사, 프런트엔드 검사, Chromium shell 테스트, 결정적 번들 검증을 포함한다. 이후 최종 정리는 이 보고서와 아키텍처 문서만 변경한다.

첫 결합 workspace 실행은 저장소가 요구하는 단일 테스트 스레드 설정을 누락하여 8,330개 통과, 실패 1개, 무시 144개에서 중단되었고 나머지 게이트는 실행하지 못했다. `vision::llmjp_vl::tests::either_patch_embedding_conv_layout_produces_the_same_features`가 최대 차이 `0.000000015832484`를 보고했다. 루트가 동일 바이너리의 해당 테스트를 직렬로 다섯 번 다시 실행하여 소스나 허용 오차 변경 없이 모두 통과했다. 올바른 게이트는 #1092에 따라 `--no-fail-fast -- --test-threads=1`을 포함하는 `make verify-test`다. 장치 전역 상태 간섭은 병렬 실행 실패의 가능한 설명이지 원인 추적으로 확정한 결과는 아니다. 처음 실패한 실행은 통과로 계산하지 않는다.

매니페스트 측정값은 초기·전체 JavaScript gzip 68,364바이트, 임베디드 자산 227,718바이트다. 새 헤드리스 모듈은 아직 scaffold에 마운트되지 않았으므로 이는 현재 scaffold의 측정값이며 최종 Models·Chat 애플리케이션 크기 예측이 아니다. 이 검사로 실행 성능 향상, GPU 추론, 실제 Safari·VoiceOver, CUDA 실행, 운영 인증 종단 간 검증을 주장하지 않는다.

사용자가 GB10 러너의 Down 상태를 확인하고 로컬 CI 통과를 전제로 실행 불가능한 GB10 작업을 건너뛰도록 명시적으로 승인했다. 이는 러너 가용성에 한정된 예외다. CI 워크플로나 브랜치 보호 설정을 바꾸지 않았으며 실행하지 않은 CUDA 검증은 통과로 보고하면 안 된다.

## 4. 변경 요약

| 영역 | 지속되는 변경 |
|---|---|
| `webui/src/api/` | 타입이 있는 동일 출처 클라이언트, 제한된 스트리밍 파서, 정식 런타임 검증과 전송 테스트. |
| `webui/src/state/` | 공유 provider·actions, 리소스 경계 리듀서, 동기화·재조정 루프와 결정적 테스트. |
| `src/server/router_lifecycle*.rs`, `router_server*.rs` | 쌍으로 전달하는 시퀀스 재생, 원자적 구독, 라우트·코디네이터 회귀 검사. |
| `docs/webui/api.yaml`, 생성 선언, 재생 fixture | 함께 변경한 재생 쿼리 계약과 안전한 숫자 커서 제약. |
| 번들 스크립트·설정·매니페스트 | 런타임 의존성 추가 없이 스키마 입력과 결정적 소스 해시 범위를 확장. |
| `docs/webui/architecture.md` | 전체 훅 계약, 추론 경로 예외, 헤드리스 통합·최신성 경계 명시. |

## 5. 학습 포인트와 후속 작업

- 재연결 커서는 마지막 수신 이벤트가 아니라 독립적으로 관측한 여러 리소스의 하한이다. 두 스냅샷 사이의 전이를 검사해야 한다.
- 중단에는 AbortController와 세대·세션 경계가 모두 필요하다. 그렇지 않으면 늦게 완료된 Promise가 로그아웃·정리 이후 상태를 되살릴 수 있다.
- 생성 타입만으로 네트워크 입력을 검증할 수 없다. fixture 전체의 런타임 검사와 독립적인 잘못된 필드 변형은 서로 보완하는 게이트다.
- 운영 시작·인증, 실제 로그인·스키마 오류 화면, 페이지 표시 계층, 브라우저 접근성 및 하드웨어 수용 검증은 후속 에픽 단위의 책임이다. 아직 구현하지 않은 `--webui` 시작 기능을 광고하지 않도록 루트 README·CLI 설치 안내는 확장하지 않았으며 최종 사용자 문서는 #1849가 담당한다.
