# 기술 보고서: PR #1871 - feat: enable model-free WebUI startup

**작성일**: 2026-09-13
**상태**: 후속 조치 필요 — `pending_host_recovery` (PR 리뷰 유지)
**언어**: Rust, TypeScript 계약 fixture, Markdown
**위험도**: 높음

## 요약

PR #1871은 이미 번들된 WebUI 셸을 `mlxcel-server --webui`와 `mlxcel serve --webui`에서 사용할 수 있는 opt-in 운영 시작 모드로 연결합니다. 모델 인자 없이 서버를 시작하고, 셸을 `{api_prefix}/webui/`에 마운트하며, typed UI API를 공통 WebUI 보안 래퍼 뒤에 배치하고, 사용자의 명시적 로드 동작 전까지 카탈로그·런타임·이벤트 관찰이 모델 로드를 유발하지 않도록 유지합니다.

## 1. 문제 정의

### 1.1 배경

WebUI 에픽에서는 스키마, 정적 번들, lifecycle coordinator, 보안 래퍼, 카탈로그 투영이 이미 단계적으로 구현됐지만, 사용자는 아직 실제 서버에서 `--webui`와 무모델 시작을 사용할 수 없었습니다. 남은 통합 과제는 llama-server 호환 의미를 깨지 않고, UI 관찰이 체크포인트 로드를 우회적으로 유발하지 않도록 하면서 이 구성 요소들을 실제 시작 경로에 조합하는 것이었습니다.

### 1.2 기존 문제점

- **모델 필수 시작 경로**: 두 서버 진입점 모두 router-mode 플래그가 없으면 시작 체크포인트를 전제로 했습니다.
- **분기된 WebUI 이벤트 의미**: 라우터와 단일 모델 이벤트 엔드포인트가 커서 검증, replay 동작, SSE 직렬화를 공유하지 않아 drift 위험이 있었습니다.
- **보안 민감 시작 정책**: 관리 UI API를 열기 전에 loopback, non-loopback, 생성 키, TLS, 명시적 root 검증을 확정해야 했습니다.
- **문서 drift**: WebUI 번들·카탈로그 문서가 통합 이후에도 운영 시작을 미래 작업으로 설명했습니다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| WebUI 시작이 예기치 않게 체크포인트를 로드하거나 네트워크에 접근함 | 높음 | 중간 |
| UI API가 공통 WebUI 보안 미들웨어를 우회함 | 높음 | 중간 |
| 라우터 runtime snapshot이 선택된 모델 설정 대신 전역 기본값을 보고함 | 중간 | 중간 |
| 오래된 문서가 운영자에게 `--webui`가 미지원이라고 안내함 | 중간 | 높음 |

## 2. 기술적 선택과 그 이유

### 2.1 기존 router/provider 권한 재사용

**컨텍스트:** WebUI에는 모델 탐색, lifecycle 작업, runtime 설정, 이벤트 스트림이 필요하지만 이를 별도 권한으로 복제하면 상태 불일치가 발생합니다.

**결정:** WebUI adapter는 `RouterPool`, `AppState`, `LifecycleCoordinator`, 기존 카탈로그 투영 캐시 위에 마운트합니다. 단일 모델 모드는 별도 provider를 등록하지 않고 기존 loaded provider를 cache-aware catalog handoff로 노출합니다.

**트레이드오프:** WebUI는 router와 provider의 제약을 그대로 따릅니다. 이후 페이지 구현은 private shortcut 대신 typed API를 사용해야 하지만, 운영 동작이 사실에 맞고 테스트 가능하게 유지됩니다.

### 2.2 셸은 공개, UI API는 인증

**컨텍스트:** 브라우저는 정적 자산을 가져올 수 있어야 하지만 control 및 observation API는 관리자 권한 표면입니다.

**결정:** 운영 앱은 정적 라우트와 UI API를 함께 `secure_webui_router`로 감싸며, 공개 셸 라우트와 private `/ui-api/v1` 라우트가 같은 Host, Origin, Fetch-Metadata, query credential, rate/body-limit 검사를 통과하도록 구성합니다.

**트레이드오프:** 운영자 키가 없는 loopback 세션은 터미널에 표시되는 생성 키가 필요합니다. 이는 URL이나 HTML에 credential을 넣지 않으면서 로컬 시작 편의성을 유지하기 위한 의도적인 선택입니다.

### 2.3 이벤트 replay 구현 공유

**컨텍스트:** 리뷰에서 단일 모델 SSE가 커서 헤더와 paired query replay 파라미터를 무시하는 단순 구독 경로를 복사한 문제가 발견됐습니다.

**결정:** PR #1871은 WebUI 이벤트 파싱과 SSE 응답 생성을 `src/server/webui/events.rs`로 추출하고, 라우터와 단일 모델 라우트가 모두 이를 호출하게 합니다. 테스트는 실제 마운트된 SSE payload를 정규화 fixture와 비교하고 paired cursor conflict 및 future sequence rejection을 검증합니다.

**트레이드오프:** helper는 WebUI feature에 묶여 있으므로 no-default-feature 빌드에서는 호출부에 명시적 cfg guard가 필요합니다. 최종 feature-off check가 이 경계를 잡아 수정했습니다.

### 2.4 표준 읽기 전용 응답과 Unicode 길이 제한 유지

Bootstrap capability의 가용성은 시작 경로 힌트가 아닌 실제 pool cache를 따릅니다. 카탈로그 진단은 파일 경로를 가린 후 스키마의 Unicode 코드 포인트 512개 제한으로 자르며, TypeScript validator도 UTF-16 코드 유닛이 아닌 코드 포인트를 셉니다. 짧은 합성 진단으로는 드러나지 않던 경계를 실제 모델 목록에서 발견했습니다.

`534563fb704619f407e4ea699482fda9c22fac98`의 읽기 전용 수정은 단일 모델 모드의 모델 작업, 다운로드, 삭제, 취소, 카탈로그 갱신에 상태 없는 거절 핸들러를 연결합니다. 각 핸들러는 모델을 변경하거나 로드하지 않고 표준 `422 unsupported`를 반환하며, 알 수 없는 operation 조회는 표준 `404`를 반환합니다. API에는 누락된 카탈로그 갱신 `422` 응답 선언만 추가하며 DTO 제한은 변경하지 않습니다. 먼저 마운트된 가짜 AppState 테스트에서 기존의 빈 `404`를 재현했습니다. 이는 CPU 기반 HTTP 검증이지 실제 단일 모델 추론 수용 검증이 아닙니다.

## 3. 변경 요약

| 범주 | 요약 |
|------|------|
| 시작 경로 | `--ui`/`--webui`가 두 바이너리에서 모델 없는 WebUI router mode를 활성화하고, 비활성 alias는 계속 허용하며, 인접한 미지원 llama.cpp UI/tool/MCP/proxy 플래그는 명확히 실패합니다. |
| 보안 | 운영 시작은 생성된 loopback 키 또는 명시적 non-loopback TLS/key 요구 사항으로 WebUI 보안 정책을 만들고 `[::1]`을 loopback으로 인식합니다. |
| 카탈로그와 런타임 | 라우터와 단일 모델 UI 카탈로그/runtime 라우트가 앱별 지속 catalog cache, 선택 entry runtime 설정, CLI/env cache root의 명시적 readable 검증을 사용합니다. |
| 계약 경계 | Cache capability는 실제 pool 상태를 따르고, 카탈로그 진단과 클라이언트 검증의 Unicode 길이 의미를 일치시키며, 단일 모델 거절 수정은 typed error envelope를 유지합니다. |
| 이벤트 | 라우터와 단일 모델 UI 이벤트가 표준 커서 파싱, replay 구독, gap 처리, SSE 직렬화를 공유합니다. |
| 문서 | WebUI bundling, catalog, architecture, llama compatibility 문서가 운영 시작 경로와 남은 미지원 인접 표면을 현재 상태에 맞게 설명합니다. |

## 4. 검증과 남은 차단 요인

검증 결과는 아래 소스 리비전에 한정됩니다. 이전 리비전의 전체 통과를 현재 변경의 전체 통과로 간주하지 않습니다.

| 리비전 / 범위 | 결과 |
|---------------|------|
| `985f4a87` 전체 로컬 게이트 | 통과: 123개 요약에서 테스트 11,236개 성공, 실패 0개, 무시 361개. Workspace all-target Clippy, 구조/계약 검사, feature-off workspace 검사를 통과했습니다. Feature-off 검사에는 경고 42개가 있었습니다. |
| `985f4a87` 실제 test-fast 바이너리 | `mlxcel-server`와 `mlxcel serve` 모두 바이너리 이동·빈 HOME·오프라인 무모델 시작 및 controlling TTY/key/port-zero authority 검증을 통과했습니다. Release 바이너리 결과는 아닙니다. |
| `985f4a87` 실제 모델 목록 | 212개 항목 중 DFlash 2개의 진단 필드 4개가 길이 제한을 초과해 strict schema 검증에 실패했습니다. 당시 실제 lifecycle 검증은 추론 전에 멈췄으며, 이 경계는 `1ff25a18`에서 수정했습니다. |
| `1ff25a18` 카탈로그와 RouterPool 수용 검증 | 212개 항목 모두 unloaded 상태를 유지하며 strict schema 검증을 통과했습니다. 실제 Llama가 467자를 스트리밍하고, drain 거절은 400을 반환했으며, worker 종료를 관찰했습니다. Granite는 “Affirmative.”를 반환했고 SIGINT 정리는 시도 1개·완료 1개를 보고했습니다. 로드된 상태의 전체 UI snapshot도 표준 계약 검증을 통과했습니다. 이는 라우터 모드 증거이지 명시적 `-m` 단일 모델 검증은 아닙니다. |
| `1ff25a18` 대상 한정 검사 | Rust 카탈로그 테스트 36개, scoped Clippy, 프런트엔드 테스트 48개, type/lint 검사, strict 계약 fixture 42개, 결정적 번들 검증이 통과했습니다. 이 변경의 독립 정확성·보안 리뷰도 통과했습니다. |
| `1ff25a18` 전체 로컬 게이트 | 변경하지 않은 `mlxcel-core`의 `dflash_round_loop_starts_at_the_configured_depth`에서 Metal `commandbufferDiscarded` / `InnocentVictim` 복구와 SIG6으로 실패했습니다. 정확히 같은 바이너리의 단독 실행은 첫 회 통과, 두 번째 회에 다시 실패했습니다. 원인은 미확정이며, 전체 통과나 일시적 오류로 간주하지 않습니다. |
| `534563fb704619f407e4ea699482fda9c22fac98` 단일 모델 거절 수정 | 유효 payload를 사용하는 CPU 기반 mounted 단일 모델 control 테스트 3개, 기존 단일 모델 SSE replay 테스트 1개, 공통 WebUI 보안 테스트 12개, strict 계약 fixture 44개, 프런트엔드 테스트 48개, type/lint, 결정적 번들 검증이 통과했습니다. Scoped Clippy와 독립 정확성·보안 변경 리뷰도 통과했습니다. |
| `534563fb` root CPU 전용 게이트 | workspace all-target Clippy, 계약 fixture 44개, 구조 검사, 포맷 및 diff 검사를 통과했습니다. GPU 테스트 결과는 아닙니다. |
| 명시적 `-m` 실제 검증과 두 release 바이너리 이동 검증 | 미실행. Mac 호스트 복구 전까지 모든 GPU 작업을 중지했으며 재부팅을 요청했습니다. |

이전 전체 실행에서는 합성 route identity 불일치(`0238c814`)와 낡은 cache fixture(`70d4e409`)도 발견했습니다. 관련 assertion을 약화하지 않고 수정했고, 이후 `985f4a87` 전체 게이트가 통과했습니다. 실제 lifecycle 중 관찰한 프로세스 RSS는 GPU 할당 해제의 증거가 아닙니다.

Maintainer의 GB10 장애 예외는 사용할 수 없는 필수 러너 검사에만 적용됩니다. 로컬 GPU 실패, 남은 release 수용 검증, 리뷰 지적은 면제하지 않으며 CUDA 실행 검증을 의미하지 않습니다. 이 작업에서 branch protection은 변경하지 않습니다.

## 5. 후속 조치

- Mac 호스트를 복구하고 실패한 GPU 게이트를 진단·재검증한 뒤 GPU 작업을 재개합니다. 복구 전 호스트에서 반복 실행하지 않습니다.
- 호스트 복구 후 최종 runtime 리비전 `534563fb704619f407e4ea699482fda9c22fac98`으로 필수 전체 검증을 다시 수행합니다.
- 명시적 `-m` 실제 단일 모델 수용 검증과 두 release 바이너리 이동·오프라인 게이트를 실행합니다. 필수 로컬 증거가 완료될 때까지 PR #1871은 리뷰 상태를 유지합니다.
- 이후 WebUI 이슈는 다운로드/삭제 adapter(#1841), rich metrics(#1847), 페이지별 workflow, 수정 UI의 Safari/VoiceOver 수용 검증을 담당합니다. 실제 GB10 CUDA 검증은 러너 복구 전까지 불가능합니다.
