# 기술 보고서: PR #1133 - refactor(server): consolidate the SSE keepalive interval and its guard

**작성일**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: Low (관측 가능한 동작 변화 없음. 모든 keepalive 값은 원래 15였고 지금도 15다)

---

## 요약

SSE keepalive 간격이 스트리밍 표면마다 하나씩, 총 세 번 선언돼 있었다. 이 숫자에 의미를 부여하는 컴파일 타임 단언, 즉 역방향 프록시의 60초 유휴 기본값보다 작아야 한다는 조건은 셋 중 하나만 이름으로 지목했다. `/v1/responses`와 Anthropic 호환 표면은 프록시 타임아웃을 넘겨 올려도 빌드가 알아채지 못했다.

우연히 값이 같은 상수 셋의 문제가 아니다. 불변식 하나에 강제 지점 하나가 있고, 표면 둘이 그 바깥에 있던 문제다. 정의를 하나로 두고, 단언을 정의 옆으로 옮겨 커버리지가 "누가 import를 기억했는가"가 아니라 "어디에 놓였는가"에서 따라 나오게 했다. `router_front.rs`의 `KeepAlive::default()` 두 곳도 원래 쓰기로 되어 있던 newtype을 거치게 했다.

---

## 1. 문제 정의

### 1.1 배경

`src/server/streaming.rs`는 keepalive를 SSE 일반의 성질로 적어 두었다. 긴 prefill은 수십 초 동안 스트림을 열어 둔 채 침묵하고, nginx와 HAProxy와 AWS ALB는 모두 유휴 타임아웃 기본값이 60초라 첫 토큰이 도착하기 전에 연결이 끊긴다. 간격이 60 미만이어야 하는 것은 스트리밍하는 모든 표면이지 특정 라우트 계열이 아니다.

스트리밍하는 표면은 셋이다. chat과 completion 라우트가 `SseKeepAlive`로, `/v1/responses`가 `ResponseSseKeepAlive`로, Anthropic 호환 messages 라우트가 `AnthropicSseKeepAlive`로 나간다. 각 newtype의 `default_for_long_prefill()` 본문은 서로 같았고, 각자 자기 파일에 선언된 상수를 읽었다.

### 1.2 기존 문제점

- **가드가 자기가 서술한 범위의 3분의 1만 덮었다.** `const _: () = assert!(SSE_KEEPALIVE_INTERVAL_SECS < 60, ...)`는 `streaming_tests.rs`에 있었고 `streaming.rs` 상수만 지목했다. 나머지 둘은 파일 전용이었고 어떤 단언도 이름을 부르지 않았다. 둘 중 아무거나 61로 바꿔도 깨끗이 컴파일됐다.
- **좁은 커버리지는 의도한 범위 제한이 아니라 흘러내림이었다.** 모듈 문서는 불변식을 SSE 일반에 대해 진술하고, 단언 메시지 자체도 "most reverse proxies"를 말하지 특정 라우트 계열을 말하지 않는다. Responses와 Anthropic 표면이 일부러 면제됐다는 근거는 코드 어디에도 없었다.
- **`router_front.rs`가 설계를 우회했다.** 두 곳이 응답을 `KeepAlive::default()`로 만들었다. axum 0.7.9에서 그 값은 15초이므로 살아 있는 버그는 아니었다. 하지만 `streaming.rs`는 내부 `KeepAlive`를 왜 비공개로 두는지 분명히 적어 두었다. "호출자가 독립적으로 어긋난 keepalive를 만드는 것을 막기 위해"다. 이 두 곳이 정확히 그 경우였다. 프록시 호환을 위해 공유 상수를 낮추면 이 둘만 아무 신호 없이 15에 남는다.
- **프로덕션 불변식이 테스트 모듈에 얹혀 있었다.** 단언이 컴파일 타임 검사라 `streaming_tests.rs`에 있어도 동작하기는 했다. 다만 상수를 감사하려는 사람이 열어 볼 생각을 하지 않을 파일에 가드의 존재가 달려 있게 됐다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|---|---|---|
| 나중에 프록시 호환 작업이 상수 하나만 낮추고 나머지 둘을 놓쳐 표면 간 값이 어긋난다 | Medium | Medium |
| Responses나 Anthropic 간격이 60을 넘겨 올라가 프록시 뒤에서 prefill 도중 스트림이 끊긴다 | High | Low |
| 공유 값을 낮춘 뒤에도 `router_front.rs`가 조용히 15초를 유지한다 | Medium | Low |

---

## 2. 기술 검토

### 2.1 이 불변식이 실제로 제약하는 대상

단언이 다루는 것은 타입이 아니라 회선이다. 서버 앞의 프록시는 어느 라우트 계열이 스트림을 만들었는지 모르고, 모두에게 같은 유휴 타임아웃을 적용한다. 따라서 가드의 올바른 범위는 "모든 SSE 표면"이고 올바른 정의 개수는 하나다. 단언을 정의 옆에 두면 그 관계가 구조가 된다. 상수를 읽는 표면은 자동으로 덮이고, 읽지 않는 표면은 공유 간격을 쓰지 않는다는 사실이 눈에 보인다.

### 2.2 newtype 셋을 그대로 두는 이유

다음 수순처럼 보이는 것, 즉 newtype 셋을 하나로 합치는 것은 틀렸고 이슈도 그렇게 적었다. 이들이 서로 다른 타입인 이유가 바로 라우트가 다른 표면의 keepalive를 붙이지 못하게 하는 데 있다. 핸들러가 받는 값은 자기가 호출한 채널 생성자에서 나온다. 합치면 컴파일 타임 보장을 내주고 아무것도 얻지 못한다. 중복은 애초에 타입에 있지 않았다. 상수와 단언에 있었고 이번에 합친 것도 그 둘뿐이다.

### 2.3 `router_front.rs`의 두 곳은 같은 결함의 반대편이다

`KeepAlive::default()`는 상수의 세 번째 사본이 아니라 상수의 부재다. axum의 기본값이 무엇이든 그대로 받는다. 그 값이 지금 15인 것은 고정된 버전에서 나온 우연이지 누가 고른 성질이 아니다. 둘 다 `SseKeepAlive::default_for_long_prefill()`을 거치게 하면 나머지 전부와 같은 정의에 묶이고, `src/server/` 안에서 `SSE_KEEPALIVE_INTERVAL_SECS`에서 파생되지 않은 SSE keepalive를 만들 방법이 남지 않는다.

### 2.4 가드가 공허하지 않음을 확인하기

통과하는 단언은 그것이 실패할 수 있는지에 대해 아무것도 알려 주지 않는다. 수용 기준이 실연을 요구했고, 실연 결과는 분명하다.

```
error[E0080]: evaluation panicked: SSE keepalive interval must be less than the 60s default used by most reverse proxies
  --> src/server/streaming.rs:77:15
```

상수를 61로 두면 그 단언에서, 이름과 메시지를 그대로 달고 빌드가 실패한다. 값은 곧바로 되돌렸다. 이 확인을 실제로 돌린 이유는, 이번 변경 전에는 나머지 두 상수에 같은 수정을 해도 깨끗이 컴파일됐다는 데 있다. 바뀐 것은 가드의 문구가 아니라 사정거리다.

---

## 3. 기술적 결정

### 3.1 단언 셋이 아니라 정의 하나

대안은 상수 셋을 그대로 두고 단언 둘을 더하는 것이었다. 고칠 줄 수는 적고 결과는 확실히 더 나쁘다. 관례로만 일치해야 하는 숫자 셋이 남고, 다음에 추가되는 표면은 다시 기본값이 미커버 상태다. 합치면 커버가 기본이 되고, 흘러내리는 쪽이 수고를 요구하게 된다.

### 3.2 단언은 공유 테스트가 아니라 정의 옆으로

테스트 모듈에 두고 나머지 두 상수를 import하는 방법도 되기는 한다. 그리고 표면이 추가될 때마다 손봐야 한다. 정의 옆에 두면 손볼 일이 없다. 모든 소비자가 그 상수를 읽으므로 모든 소비자가 덮인다.

### 3.3 `router_front.rs`는 상수가 아니라 newtype을 거친다

두 곳에 `KeepAlive::new().interval(Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS))`를 인라인으로 쓰면 기준의 문구는 만족하면서 생성 식 사본을 둘 더 만든다. `SseKeepAlive::default_for_long_prefill()`을 부르면 그 식이 사는 유일한 자리를 재사용한다.

### 3.4 모듈 문서가 숫자를 다시 적지 않는다

모듈 문서에는 "The keepalive interval is 15 seconds"라고 적혀 있었다. 어떤 단언도 닿지 못하는 산문에 값이 흘러내릴 네 번째 자리가 있었던 셈이다. 이제는 상수를 이름으로 가리키고 그것이 공유된다는 사실을 적는다. 독자에게 필요한 정보는 그쪽이다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 5 |
| 제거한 상수 정의 | 2 |
| 제거한 `KeepAlive::default()` 호출부 | 2 |
| 동작 변경 | 0 |

### 영역별 변경

**`src/server/streaming.rs`**
- `SSE_KEEPALIVE_INTERVAL_SECS`를 모든 SSE 표면의 단일 정의로 문서화.
- `const _: () = assert!(... < 60, ...)`를 `streaming_tests.rs`에서 정의 바로 아래로 이동.
- 모듈 문서가 `15`를 다시 적는 대신 상수를 가리키고, 불변식이 이제 모든 표면을 덮는다는 사실을 기록.

**`src/server/streaming_responses.rs`, `src/server/streaming_anthropic.rs`**
- 파일 전용 `const KEEPALIVE_INTERVAL_SECS: u64 = 15;` 삭제, 공유 상수 import. newtype 자체는 손대지 않았다.

**`src/server/streaming_tests.rs`**
- 단언과 이제 쓰이지 않는 import 제거. 불변식이 어디로 왜 옮겨 갔는지 기록하는 주석으로 대체.

**`src/server/router_front.rs`**
- `KeepAlive::default()` 두 곳이 `SseKeepAlive::default_for_long_prefill()`을 거친다. `KeepAlive` import 제거.

---

## 5. 검증과 후속

### 통과

- GB10에서 `cargo test --profile test-fast --features cuda --lib server::streaming`: 25개 통과, 0개 실패.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`.
- `cargo fmt --all -- --check`.
- 위에 인용한 `61` 빌드 실패 확인. 확인 후 되돌렸다.

### 덮지 않은 것

- 수용 기준은 `--features metal,accelerate`를 지목한다. 이 장비는 Linux CUDA 박스라 그 피처 조합을 빌드할 수 없다. 해당 코드는 어떤 피처 게이트도 닿지 않는 백엔드 무관 SSE 배관이므로, CUDA 실행이 대체재가 아니라 동등한 증거다.
- 실제 프록시 시험은 없다. 나가는 프레임에 바뀐 것이 없으므로 회선에서 새로 관찰할 것도 없다. 값은 이미 프로덕션에 있는 것과 동일하다.

### 후속

- #1107이 이 중복의 나머지 절반, 즉 `Sse::new(..).keep_alive(..)` 부착 식 셋을 합치며 이 변경 위에 바로 얹힌다.
