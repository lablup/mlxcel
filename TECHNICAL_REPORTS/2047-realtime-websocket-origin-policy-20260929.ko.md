# 기술 보고서: PR #2047 - /v1/realtime 업그레이드에 CORS origin 정책 적용

**날짜**: 2026-09-29

**상태**: 구현 완료, fake 엔진 테스트로 검증, 머지 대기.

**언어**: Rust

**위험도**: 낮음. 변경은 realtime 라우트와 `CorsPolicy` 메서드 하나에 한정된다. 기본 `--cors-origins *` 동작은 바뀌지 않는다.

## 요약

`GET /v1/realtime`은 `--cors-origins`나 `--allowed-origins` 설정과 무관하게 모든 `Origin`의 WebSocket 업그레이드를 받아들였다. 브라우저는 WebSocket 업그레이드에 CORS를 적용하지 않으므로, 어떤 웹 페이지든 소켓을 열어 하나뿐인 VoiceChat 세션 슬롯을 차지하고 모든 이벤트를 읽을 수 있었다(cross-site WebSocket hijacking). 이제 라우트가 설정된 정책으로 `Origin`을 검사하고, 업그레이드 전에 `403`으로 응답한다(이슈 #2042).

## 1. 문제 정의

`cors_middleware`는 응답에 `Access-Control-Allow-Origin`을 붙이기만 하고 요청을 거부하지 않으며, 라우터 모드 서브 앱에서는 아예 실행되지 않는다. 브라우저는 이 헤더와 상관없이 WebSocket 핸드셰이크를 완료하므로, `--cors-origins localhost`나 `--allowed-origins https://app.example.com`을 설정한 운영자도 `/v1/realtime`에 대해서는 브라우저 제한이 없었다. 노출되는 경우는 `--api-key` 없이 실행된 서버(브라우저는 `new WebSocket`에 `Authorization`을 설정할 수 없다)에 사용자가 방문한 페이지가 접근하는 경우다. 영향은 유일한 세션 슬롯 점유와 연산 자원 사용이다.

## 2. 변경 요약

- **정책** (`server::cors`): `CorsPolicy::permits_websocket_origin(Option<&HeaderValue>)`. `Origin`이 없으면 허용한다. 있으면(`null` 포함) origin 규칙을 만족해야 한다: `Wildcard`는 모든 값, `Localhost`는 localhost 호스트(기존 `origin_is_localhost`), `Literal`은 바이트 단위 일치, `AllowList`는 목록 포함. `credentials`는 관여하지 않는다.
- **라우트** (`server::routes::realtime`): 비공개 `RealtimeRouteState`를 쓰는 `realtime_router(engine, cors)`. 핸들러는 origin을 먼저 검사하고, 거부 시 `invalid_request_error` JSON 본문과 함께 `403`을 반환하며 `warn!` 로그를 한 번 남긴다. 예약은 잡지 않는다.
- **연결** (`server::app::build_routes`): `Arc::new(config.cors_policy.clone())`을 전달한다. 검사가 라우트에 있으므로 `create_app_without_cors`(라우터 모드 서브 앱)에서도 적용된다.
- **문서**: `docs/nemotron-voicechat.md`에 한 문장 추가.

## 3. 기술적 결정

### `cors_middleware`가 아닌 라우트에서 검사

미들웨어에서 거부하면 llama-server b10621을 따르는 HTTP CORS 동작이 바뀌고, 라우터 모드 서브 앱에서는 미들웨어가 생략된다. 검사가 필요한 라우트 하나에만 둔다.

### 업그레이드 검증보다 origin 검사를 먼저

이슈는 핸들러의 첫 추출자로 `ws: WebSocketUpgrade`를 제시했다. 실제 구현은 `Result<WebSocketUpgrade, WebSocketUpgradeRejection>`을 마지막 추출자로 받고 `Origin`을 먼저 검사한다. 핸드셰이크가 잘못된 경우에도 허용되지 않은 origin은 거부되며, 프로세스 내 앱 테스트가 hyper 업그레이드 핸들 없이 `create_app`을 통해 `403`을 확인할 수 있다. 허용된 origin에 잘못된 핸드셰이크는 여전히 axum 자체의 거부 응답을 받는다.

### Origin 없는 요청 허용, API 키 검사가 바깥

브라우저는 업그레이드 시 항상 `Origin`을 보내지만 `websocat`, 예제 클라이언트, 테스트는 보내지 않는다. 헤더가 없는 요청을 허용해도 브라우저 경우의 보호는 약해지지 않는다. API 키 레이어가 라우트를 감싸므로 두 검사 모두 실패하면 이전처럼 `401`을 받는다.

## 4. 검증

- `websocket_origin_matrix` 단위 테스트: 모든 정책 변형을 origin 없음, 허용, 비허용, `null`, 비 UTF-8 값에 대해 credentials 켜짐과 꺼짐 모두로 검사.
- `create_app`과 `create_app_without_cors`에 대한 앱 수준 테스트: 비허용 origin은 `403`, 허용 또는 없음은 `403` 아님, API 키가 없으면 `401`.
- `tests/realtime_ws.rs`: `start_server_with_policy` 추가. 비허용 origin은 HTTP 403으로 핸드셰이크 실패, 이후 허용 origin은 첫 시도에 `session.created`를 받고(거부된 시도가 예약을 잡지 않았음을 확인) `Origin` 없는 클라이언트는 해제 후 세션을 받는다. localhost 정책은 `https://localhost.evil.com`을 거부하고 `http://localhost:3000`을 허용한다. 기본 정책은 모든 origin을 허용한다.
- `cargo fmt --check`, `cargo clippy --release -p mlxcel --lib --tests --examples -D warnings`, 세 계약 테스트, `cargo test --release -p mlxcel --lib server::`(3270 통과, 0 실패)가 로컬에서 통과했다. GB10 러너가 중단되어 GitHub CI는 기다리지 않았다.

## 5. 알려진 한계

- 실제 체크포인트로는 실행하지 않았다. fake 엔진 테스트가 같은 라우트와 핸드셰이크를 검증한다.
- 라우터 모드는 현재 WebSocket 업그레이드를 프록시하지 않으므로 라우터 수준 변경은 필요 없었다.
