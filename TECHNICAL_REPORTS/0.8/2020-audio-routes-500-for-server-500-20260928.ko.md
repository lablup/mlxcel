# 기술 리포트: PR #2020, audio 라우트가 서버 측 실패에 500을 반환하도록 수정

**작성일**: 2026-09-28
**상태**: 완료
**언어**: Rust
**위험도**: Low

## 요약

`src/server/routes/audio.rs`의 여섯 실패 경로(`audio_speech`, `compat_transcribe`, `transcribe`의 워커 패닉 처리, 트랜스크립션 스트림 및 바이너리 오디오 응답 경로의 응답 빌더 실패, `AudioModelError::Inference` 매핑)가 서버 측 실패임에도 HTTP 400을 반환하고 있었다. 각 지점이 `.status`를 덮어쓰지 않고 `ErrorResponse::new(message, "server_error")`만 호출해 생성자의 기본값인 `BAD_REQUEST`가 그대로 새어 나왔기 때문이다. 형제 라우트인 embeddings와 rerank는 동일한 종류의 실패에서 이미 500을 반환하고 있었다. 이번 PR은 여섯 지점 전부를 동반 PR #2015(이슈 #1690)에서 추가한 `ErrorResponse::internal_server_error(message)` 생성자로 전환하고, 기존 테스트에서 검증되지 않았던 `Inference` 분기의 상태 코드를 고정한다.

## 1. 문제 정의

### 1.1 배경

`ErrorResponse::new`는 `status`를 기본값 `BAD_REQUEST`(400)로 설정한다. 서버 측 장애를 알리려면 호출자가 `.status`를 명시적으로 덮어써야 한다. rerank와 embeddings 라우트는 동등한 실패 클래스(워커 패닉, 잘못된 형식의 프로바이더 결과)에서 이미 이를 수행하고 있었지만, audio 라우트에는 애초에 이 덮어쓰기 줄 자체가 없었다.

### 1.2 기존 문제점

- **문제 1**: 서버 측 실패(패닉한 `spawn_blocking` 작업, SSE 또는 바이너리 바디를 만드는 `Response::builder()` 실패, `AudioModelError::Inference`)가 `"type": "server_error"`를 담은 채 HTTP 400으로 클라이언트에 보고되고 있었다. 상태 코드가 에러 타입과 모순되어, 상태 코드로 재시도나 알림 동작을 결정하는 프록시나 클라이언트가 이를 호출자의 잘못으로 취급할 수 있었다.
- **문제 2**: 기존 테스트 `model_error_maps_kinds_and_inference`는 `Inference` 분기에서 `error_type == "server_error"`만 검증하고 상태 코드는 검증하지 않아, `KindNotLoaded`(501), `QueueFull`(503), `Timeout`(504) 분기와 달리 올바른 동작이 테스트로 고정되어 있지 않았다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|--------|------------|
| 5xx를 기준으로 서버 상태를 감지하는 모니터링/알림 시스템이 audio 워커 패닉과 추론 실패를 전혀 감지하지 못함(4xx로 나타나므로) | Medium | 수정 전에는 확실히 발생; 이번 수정으로 해소 |

## 2. 기술적 검토 사항

### 2.1 코드 품질 관점

- **테스트 커버리지**: `assert_eq!(inference.status, StatusCode::INTERNAL_SERVER_ERROR)` 검증 1건 추가로, 기존 `model_error_maps_kinds_and_inference` 테스트가 `AudioModelError`의 네 변형 모두를 커버하게 되었다. `server::routes::audio` 모듈 전체(16개 테스트)가 통과한다.
- **코드 복잡도**: 변화 없음. 각 지점은 여전히 단일 표현식이며, `ErrorResponse::new` 대신 공유 생성자를 호출하도록만 바뀌었다.

### 2.2 호환성 및 의존성 관점

- **호환성 파괴 여부**: 좁은 의미에서는 있음. 여섯 응답 바디가 동일한 본문(메시지와 `"type": "server_error"`는 그대로)에 대해 이제 HTTP 400 대신 500을 반환한다. 에러 타입이 아니라 이 경로들에 대해 400을 특정해 분기하던 클라이언트나 프록시는 새 상태 코드를 관측하게 된다. 이는 의도된 수정이다: 이들은 서버 측 실패이므로 500이 올바른 상태 코드다.
- **신규 의존성**: 없음.

## 3. 기술적 선택과 그 이유

### 3.1 인라인 수정 대신 #1690 생성자 재사용

**컨텍스트**: 이슈는 두 가지 선택지를 제시했다: 각 지점에서 직접 `INTERNAL_SERVER_ERROR`를 설정하거나, 동반 이슈 #1690이 먼저 머지되면 그 공유 생성자를 사용하는 것.

**선택 이유**: 이 체인에서 #1690(PR #2015)이 먼저 머지되어 `ErrorResponse::internal_server_error`를 이미 사용할 수 있었다. 이를 사용하면 audio 라우트가 rerank, embeddings, gcp_compat과 일관성을 유지하고, 애초에 이 결함을 낳았던 두 줄짜리 패턴을 다시 도입하는 것을 피할 수 있다.

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/server/routes/audio.rs`** (`AudioModelError::Inference` 분기)
```rust
// 변경 전
AudioModelError::Inference(message) => ErrorResponse::new(
    format!("audio model inference failed: {message}"),
    "server_error",
),

// 변경 후
AudioModelError::Inference(message) => {
    ErrorResponse::internal_server_error(format!("audio model inference failed: {message}"))
}
```

**파일: `src/server/routes/audio.rs`** (테스트)
```rust
let inference = audio_model_error_response(AudioModelError::Inference("boom".into()));
assert_eq!(inference.error.error_type, "server_error");
assert!(inference.error.message.contains("boom"));
assert_eq!(inference.status, StatusCode::INTERNAL_SERVER_ERROR); // 신규
```

**변경 이유**: 형제 embeddings/rerank 관례와 일치시키고, 테스트가 검증하던 내용과 라우트가 실제로 반환하던 값 사이의 간극을 없앤다.

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-------|
| 변경된 파일 | 1 |
| 추가된 줄 | +9 |
| 삭제된 줄 | -9 |
| 추가된 테스트 | 1건 (기존 테스트 내 검증) |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|-------|---------|
| 버그 수정 | 호출 지점 6개 | 진짜 서버 측 실패에 대한 HTTP 상태를 400에서 500으로 수정 |
| 코드 품질 | 검증 1건 | 기존 테스트에서 `Inference` 분기의 상태 코드를 고정 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|---------|
| `93e3417` | fix | Audio routes answer 500 for server-side failures |

### 관련 PR/이슈

- PR #2015 / 이슈 #1690: 이번 PR이 사용하는 `ErrorResponse::internal_server_error` 생성자를 추가했다.

## 8. 후속 조치

### 완료 필요

- [ ] 없음.

### 모니터링 필요

- 일반적인 배포 관찰 외 특별히 필요 없음. 이 여섯 가지 audio 실패 모드를 이전에 4xx로 집계하던 대시보드나 알림이 있다면 이제 올바르게 5xx로 집계된다.
