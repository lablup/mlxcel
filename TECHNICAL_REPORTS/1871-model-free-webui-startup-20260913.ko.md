# 기술 보고서: PR #1871 - feat: enable model-free WebUI startup

**작성일**: 2026-09-13
**상태**: 완료
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
| WebUI 시작이 예기치 않게 체크포인트를 로드하거나 네트워크를触함 | 높음 | 중간 |
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

## 3. 변경 요약

| 범주 | 요약 |
|------|------|
| 시작 경로 | `--ui`/`--webui`가 두 바이너리에서 모델 없는 WebUI router mode를 활성화하고, 비활성 alias는 계속 허용하며, 인접한 미지원 llama.cpp UI/tool/MCP/proxy 플래그는 명확히 실패합니다. |
| 보안 | 운영 시작은 생성된 loopback 키 또는 명시적 non-loopback TLS/key 요구 사항으로 WebUI 보안 정책을 만들고 `[::1]`을 loopback으로 인식합니다. |
| 카탈로그와 런타임 | 라우터와 단일 모델 UI 카탈로그/runtime 라우트가 앱별 지속 catalog cache, 선택 entry runtime 설정, CLI/env cache root의 명시적 readable 검증을 사용합니다. |
| 이벤트 | 라우터와 단일 모델 UI 이벤트가 표준 커서 파싱, replay 구독, gap 처리, SSE 직렬화를 공유합니다. |
| 문서 | WebUI bundling, catalog, architecture, llama compatibility 문서가 운영 시작 경로와 남은 미지원 인접 표면을 현재 상태에 맞게 설명합니다. |

## 4. 검증

- 선택 runtime 설정, paired SSE replay, invalid cursor, 명시적 model-store root, bracketed IPv6 loopback 분류, 단일 모델 mounted SSE replay fixture에 대한 targeted WebUI route/startup 테스트가 통과했습니다.
- `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`가 통과했습니다.
- `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python`가 41개 fixture로 통과했습니다.
- `make verify-llama-compat verify-versions verify-kernel-dtype-keys`가 통과했습니다.
- `cargo check --no-default-features --features metal,accelerate,surgery --lib --tests`는 기존 no-WebUI unused warning만 남기고 통과했습니다.
- 필수 GB10 CUDA CI는 러너가 down 상태라 실행하지 않았습니다. maintainer는 local validation으로 진행하고, 해당 unavailable required job은 root가 merge exception 경로로 처리하는 것을 승인했습니다.

## 5. 후속 조치

- 머지 전 독립 구현 리뷰와 보안 리뷰를 완료해야 합니다.
- root는 에픽에서 요구하는 broad workspace/full production binary gate와 serialized real-model acceptance를 실행해야 합니다.
- 이후 WebUI 이슈는 page-level chat, download/removal, rich metrics, revised UI의 Safari/VoiceOver acceptance, 러너 복구 후 실제 GB10 CUDA 검증을 계속 담당합니다.
