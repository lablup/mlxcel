# 기술 리포트: PR #2015, ErrorResponse::internal_server_error 추가

**작성일**: 2026-09-28
**상태**: 완료
**언어**: Rust
**위험도**: Low

## 요약

`ErrorResponse`는 반환하는 5xx 계열 상태 코드 중 500만 빼고 전용 생성자를 이미 갖추고 있었다: `service_unavailable`(503), `gateway_timeout`(504), `not_supported`/`not_implemented`(501). 500 케이스만 `rerank.rs`, `embeddings.rs`, `gcp_compat.rs`의 여덟 개 지점에서 `ErrorResponse::new(message, "server_error")`를 호출한 뒤 별도로 `response.status = StatusCode::INTERNAL_SERVER_ERROR`를 대입하는 방식으로 직접 작성되어 있었다. 이번 PR은 형제 생성자들과 나란히 `ErrorResponse::internal_server_error(message)`를 추가하고 여덟 지점 전부를 이 생성자로 전환한다. 동작 변화는 없다: 모든 지점이 기존과 동일한 상태 코드와 메시지 텍스트를 유지한다.

## 1. 문제 정의

### 1.1 배경

`ErrorResponse::new`는 `status`를 기본값 `StatusCode::BAD_REQUEST`(400)로 설정한다. 400이 아닌 모든 에러 경로는 생성 후 `status`를 명시적으로 덮어써야 하며, 프로젝트는 이미 상태 코드마다 에러 타입, 메시지, 상태 덮어쓰기를 하나로 묶은 이름 있는 생성자들을 갖추고 있었다. 500만 유일하게 두 줄짜리 수작업 방식으로 남아 있었다.

### 1.2 기존 문제점

- **문제 1**: 호출 지점에서 `server_error` 메시지만 설정하고 뒤이은 `response.status = StatusCode::INTERNAL_SERVER_ERROR` 줄을 빠뜨릴 수 있어, 서버 측 장애가 `new`의 기본값인 400으로 조용히 격하될 위험이 있었다. 이것이 정확히 이슈 #1695가 audio 라우트에서 지적하는 결함으로, audio.rs는 애초에 이 두 번째 줄 자체가 없었다.
- **문제 2**: 여덟 지점에 중복된 코드가 있어, 500 생성 방식에 어떤 변경(예: `code` 필드 추가)이 생겨도 여덟 곳을 모두 고쳐야 했다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|--------|------------|
| 수작업 패턴을 그대로 두면 새 호출 지점에서 동일한 400-for-500 결함이 재발할 수 있음 | Medium | Medium (audio.rs에서 이미 한 번 발생) |

## 2. 기술적 검토 사항

### 2.1 코드 품질 관점

- **테스트 커버리지**: 변화 없음. 기존 `rerank`, `embeddings` 라우트 테스트 스위트가 이미 전환된 경로에서 `StatusCode::INTERNAL_SERVER_ERROR`를 검증하고 있다(`non_finite_scores_return_500`, `non_finite_embeddings_return_500`, `provider_errors_map_to_the_shared_status_codes`, `error_mapping_matches_audio_routes`). 전부 수정 없이 통과했으며, 이는 리팩터링이 동작을 그대로 보존했음을 확인해 준다.
- **코드 복잡도**: 감소. 전환된 각 지점은 `let mut response = ErrorResponse::new(...)` 후 상태 대입(세 지점은 추가로 명시적인 `response` 반환/`return response.into_response()`)을 거치던 것이 단일 표현식으로 축약되었다.
- **기술 부채**: 감소. `rerank.rs`, `embeddings.rs`, `gcp_compat.rs` 각 파일에서 마지막 남은 `StatusCode::INTERNAL_SERVER_ERROR` 참조가 제거되면서 `axum::http::StatusCode` import가 죽은 코드가 되었고, 세 파일 모두에서 삭제했다.

### 2.2 호환성 및 의존성 관점

- **호환성 파괴 여부**: 없음. 와이어 포맷, 상태 코드, 메시지 텍스트는 변경되지 않았다.
- **신규 의존성**: 없음.

## 3. 기술적 선택과 그 이유

### 3.1 생성자 배치와 문서 주석

**컨텍스트**: 기존 이름 있는 생성자 네 개(`service_unavailable`, `gateway_timeout`, `not_supported`, `not_implemented`)는 `response.rs`에 함께 모여 있고, 각각 어떤 라우트가 사용하는지 설명하는 짧은 문서 주석이 붙어 있다.

**선택 이유**: `internal_server_error`는 `gateway_timeout` 바로 뒤에 배치해 5xx 계열 생성자들을 인접하게 유지했고, 호출 지점을 나열하는 대신(패닉한 워커 작업, 잘못된 형식의 프로바이더 결과, 응답 빌더 오류 등) 이 생성자가 대표하는 장애 유형을 설명하는 동일한 한 문단 스타일의 문서 주석을 붙였다. 호출 지점은 #1695로 더 늘어날 예정이라 목록을 나열하면 금방 낡은 정보가 되기 때문이다.

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/server/types/response.rs`**
```rust
// 변경 후
pub fn internal_server_error(message: impl Into<String>) -> Self {
    Self {
        error: ErrorDetail {
            message: message.into(),
            error_type: "server_error".into(),
            code: None,
        },
        status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
    }
}
```

**파일: `src/server/routes/rerank.rs`** (전환된 네 지점 중 하나)
```rust
// 변경 전
let mut response = ErrorResponse::new(
    format!("rerank inference failed: {message}"),
    "server_error",
);
response.status = StatusCode::INTERNAL_SERVER_ERROR;
response

// 변경 후
ErrorResponse::internal_server_error(format!("rerank inference failed: {message}"))
```

**변경 이유**: 메시지만 설정하고 상태를 빠뜨릴 가능성 자체를 없애고, 이미 503/504/501에 쓰이던 형제 생성자 관례와 일치시킨다.

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-------|
| 변경된 파일 | 4 |
| 추가된 줄 | +35 |
| 삭제된 줄 | -45 |
| 추가된 테스트 | 0 (기존 테스트가 전환된 경로를 이미 커버) |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|-------|---------|
| 코드 품질 | 호출 지점 8개 + 신규 생성자 1개 | 중복된 500 생성 패턴을 이름 있는 생성자 하나로 축약 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|---------|
| `2798675` | refactor | Add ErrorResponse::internal_server_error |

## 8. 후속 조치

### 완료 필요

- [ ] 없음. 이슈 #1695(audio 라우트가 서버 측 실패에도 400을 반환하는 문제)가 이 PR의 직접적인 후속 작업이며 이 생성자를 사용하게 된다.

### 향후 개선 사항

- `audio.rs`의 `server_error` 지점 여섯 곳(이슈 #1695)은 현재 상태 덮어쓰기 자체가 없으며, `internal_server_error`의 다음 자연스러운 사용처다.
