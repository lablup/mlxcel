# 기술 보고서: PR #1136 - refactor(server): attach the SSE keepalive through one constructor

**작성일**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: Low (관측 가능한 동작 변화 없음. 모든 라우트가 같은 프레임을 같은 주기로 내보낸다)

---

## 요약

라우트 핸들러 다섯이 각자 같은 세 줄짜리 꼬리로 SSE keepalive를 손수 붙이고 있었다. 이를 빠뜨린 여섯 번째 라우트가 생겨도 컴파일되고 전체 스위트를 통과한다. `TODO`는 이 구멍을 메우기 위해 axum 통합 테스트 하네스를 요청하고 있었다.

하네스는 맞는 도구가 아니고, 그 판단 근거가 이번 변경의 알맹이다. 회귀할 수 있는 성질은 "모든 라우트가 keepalive를 붙이는가"이고, 통합 테스트는 누군가 테스트를 써 준 라우트에 대해서만 답한다. 라우트마다 테스트 하나면 복붙 다섯 자리를 지키는 테스트 다섯이다. 같은 중복을 테스트 모듈로 옮긴 것일 뿐, 여섯 번째 라우트에는 여전히 눈이 멀어 있다. 중복을 없애면 그 성질이 거짓이 될 능력 자체가 사라진다.

이제 `streaming::sse_response`가 이 스트림들을 `Response`로 바꾸는 유일한 길이고, keepalive를 값으로 받는다. 트리 전체에 `Sse::new` 하나와 `.keep_alive` 하나가 남았고, 호출부 일곱이 그리로 지난다.

---

## 1. 문제 정의

### 1.1 배경

`src/server/streaming_tests.rs`에는 알려진 커버리지 구멍을 기록한 `TODO`가 있었다. `Sse::new(stream).keep_alive(...)`를 종단으로 구동해 원시 SSE 프레임을 읽는 테스트가 없다는 내용이다. 단위 수준의 `payload_channel` 테스트는 axum의 `KeepAlive` 계층에 닿지 못하고, 테스트 인프라를 늘리지 않으려고 건너뛰었다고 적혀 있었다.

그 뒤에는 핸들러 다섯이 각자 같은 꼬리를 달고 있었다.

```rust
Sse::new(stream)
    .keep_alive(keepalive.into_inner())
    .into_response()
```

`routes/chat.rs`, `routes/completions.rs`, `routes/native_completion.rs`, `routes/responses.rs`, `routes/anthropic.rs`다. 스트림 타입도, keepalive도, 순서도 라우트마다 다르지 않았다.

### 1.2 기존 문제점

- **불변식이 각자 기억하는 행위 다섯 개로 유지되고 있었다.** 라우트가 keepalive를 붙이도록 강제하는 장치가 타입 체계에 없었다. `Sse::new(stream).into_response()`를 반환하는 핸들러도 멀쩡히 컴파일되고, 역방향 프록시가 prefill 도중 끊어 버릴 스트림을 서빙한다.
- **제안된 수정은 중복을 제도화했을 것이다.** 라우트당 `tower::ServiceExt::oneshot` 테스트 하나면, 라우트를 추가할 때마다 추가해야 하는 테스트 다섯이 라우트를 추가할 때마다 써야 하는 자리 다섯을 지키는 구조다. 실패 양상은 똑같다. 여섯 번째 라우트를 추가하고 잊는다. 테스트 모듈은 유지보수 의무를 얻고 새 보장은 얻지 못한다.
- **`TODO`의 절반은 업스트림을 시험하는 것이었다.** "스트림이 유휴일 때 axum이 주석 프레임을 내보내는가"는 axum `KeepAlive`의 성질이지 mlxcel의 성질이 아니다. 업스트림이 이미 덮고 있다. 여기서 재현하면 이 코드베이스에 대해 아무것도 말해 주지 않는 결과를 위해 인프라만 늘린다.
- **`TODO`가 말한 비용은 이미 낮아져 있었고, 결과적으로 상관이 없었다.** `tower = { version = "0.4", features = ["util"] }`가 이미 직접 의존성이고 `util`이 바로 `ServiceExt::oneshot`을 주는 피처다. 하네스에 새 의존성은 필요 없었다. 계획을 반려하기 전에 확인할 값어치는 있고, 다른 근거로 반려되는 순간 무관해진다.
- **`router_front.rs`에 손으로 조립한 자리가 둘 더 있었다.** `src/server/routes/` 바깥이라 이슈 문구 밖이지만 같은 구성이고, 남겨 두면 "부착 지점 하나"가 트리가 아니라 디렉터리의 성질이 된다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|---|---|---|
| keepalive 없는 라우트가 추가되어 긴 prefill 중 프록시 뒤에서 스트림이 끊긴다 | High | Medium |
| 라우트별 테스트 다섯이 추가된 뒤 라우트 목록과 어긋난다 | Medium | High |
| 공유 생성자가 다른 스트림 타입이 정말 필요한 미래 라우트를 과하게 제약한다 | Low | Low |

---

## 2. 기술 검토

### 2.1 `TODO`의 두 절반은 서로 다른 답을 원한다

성질을 쪼개면 판단이 자명해진다. "axum이 연결을 살려 두는가"는 axum의 몫이다. "모든 라우트가 그렇게 요청하는가"는 이쪽 몫이고, 시험 가능한 성질이라기보다 구조적 성질이다. 테스트는 자기가 아는 라우트만 표집할 수 있고, 타입은 아직 쓰이지 않은 것까지 포함해 전부를 제약할 수 있다. 미래 코드에 대한 전칭 명제를 다루는 올바른 도구는 컴파일러다.

### 2.2 무엇이 "빠뜨리기"를 표현 불가능하게 만드는가

`sse_response(stream, keepalive)`는 keepalive를 값으로 받고, newtype의 내부 `KeepAlive`는 비공개다. 따라서 이 스트림들에서 `Response`로 가는 공개 경로 중 여기를 지나지 않는 것이 없다. keepalive를 건너뛰려는 라우트는 `Sse`를 직접 만들어야 하고, 그러려면 import를 해야 하며, 그것은 이제 어느 라우트도 하지 않는 일로 눈에 띈다. 보장의 내용은 "쓸 수 없다"가 아니라 "빠뜨려서는 될 수 없다"이다.

### 2.3 원시 `KeepAlive`로 가는 마지막 공개 경로를 닫았다

newtype마다 `into_inner()` 접근자가 있었고 그중 둘은 `pub mod` 안의 `pub`이었다. 모든 라우트가 `sse_response`를 지나게 되자 이들의 유일한 호출자는 자기 `IntoKeepAlive` impl이 되었고, 동시에 이 모듈 바깥 코드가 맨 `KeepAlive`를 얻어 `Sse`를 손으로 조립할 수 있는 마지막 통로였다. 셋 다 제거했고 트레이트 impl이 `self.0`을 직접 돌려준다. 리뷰에서는 선택 사항으로 제기됐다. 취할 값어치가 있다. 이 PR의 핵심 주장이 손 조립에 도달할 수 없다는 것인데, 공개 접근자를 열어 두면 바로 그 지점에서 주장이 구조가 아니라 관례로만 참이 되기 때문이다.

### 2.4 트레이트를 일부러 최소로 두었다

`IntoKeepAlive`는 메서드가 하나이고 존재 이유도 하나다. 서로 다른 타입 셋이 함수 하나에 닿게 하는 것. newtype을 통합하지 않으며, 그것이 핵심이다. #1105는 라우트가 다른 표면의 keepalive를 붙이지 못해야 한다고 요구했고 그 성질은 그대로다. 핸들러가 `sse_response`에 넘기는 값은 자기 채널 생성자가 돌려준 것이기 때문이다. 트레이트는 생성자가 받을 수 있는 것의 폭을 넓히지, 특정 라우트가 넘길 수 있는 것의 폭을 넓히지 않는다.

### 2.5 컴파일러가 결과를 확인해 주었다

꼬리 다섯을 지우자 라우트 파일 다섯 곳에서 `sse::Sse`가 미사용이 되었고, clippy가 `-D warnings` 아래 각각을 이름으로 짚으며 빌드를 실패시켰다. 이것은 잡일이 아니라 증거로 기록할 값어치가 있다. 이제 어떤 라우트도 SSE 응답을 스스로 만들지 않는다는 컴파일러의 진술이다. import는 제거했다.

최종 상태는 각각 한 줄로 확인된다. `grep -rn "Sse::new(" src/`와 `grep -rn "\.keep_alive(" src/`가 각각 정확히 한 줄, 둘 다 `streaming.rs`를 돌려주고, 호출부 일곱이 그리로 지난다.

---

## 3. 기술적 결정

### 3.1 테스트 다섯이 아니라 생성자 하나

위에 적었다. 짧게 줄이면, 테스트는 표집하고 타입은 전칭한다. 이 가드가 막으려는 회귀는 아직 존재하지 않는 코드에 관한 것이라 테스트가 닿을 수 없다.

### 3.2 통합된 newtype 하나가 아니라 공유 트레이트

newtype 셋을 합쳐도 중복은 사라졌을 것이고 #1105가 지키는 보장을 대가로 냈을 것이다. 중복은 애초에 타입에 없었다. 부착 식에 있었고, 합친 것도 그것뿐이다.

### 3.3 범위 밖인데도 `router_front.rs`를 포함한 이유

이슈는 `src/server/routes/`를 지목한다. 이 두 자리는 `src/server/`에 있고 `sse_channel`에서 오지 않으며 newtype을 직접 만든다. 남겨 두면 기준은 만족하면서 미래의 수정이 복사해 갈 수 있는 손 조립 `Sse::new` 둘이 남는다. 포함해야 "부착 지점 하나"가 트리의 성질이 된다.

### 3.4 `TODO`를 지우기만 하지 않고 대체했다

그냥 지우면 근거가 사라지고, 다음 독자가 같은 커버리지 구멍을 발견해 같은 하네스를 다시 제안한다. 대체 주석은 위험이 구조적으로 제거되었다는 것과, `TODO`의 나머지 절반이 업스트림을 시험한다는 것을 기록한다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 10 |
| 제거한 손 조립 부착 지점 | 7 |
| 남은 부착 지점 | 1 |
| 제거한 원시 `KeepAlive` 공개 접근자 | 3 |
| 신규 공개 API | 0 (추가된 둘 다 `pub(crate)`) |
| 동작 변경 | 0 |

### 영역별 변경

**`src/server/streaming.rs`**
- `IntoKeepAlive` 트레이트 신설과 `SseKeepAlive`용 impl.
- `sse_response<S>(stream, keepalive) -> Response` 신설. 유일한 부착 지점이다.

**`src/server/streaming_responses.rs`, `src/server/streaming_anthropic.rs`**
- `IntoKeepAlive` impl 각각 하나. 채널 생성자 문서가 손 조립을 서술하는 대신 `sse_response`를 가리킨다.

**`src/server/routes/{chat,completions,native_completion,responses,anthropic}.rs`**
- 각 꼬리가 `sse_response(stream, keepalive)`가 되고, 미사용이 된 `sse::Sse` import를 제거.

**`src/server/router_front.rs`**
- 두 자리 모두 `sse_response`를 거치고, `Sse` import 제거.

**`src/server/streaming_tests.rs`**
- `TODO`를 위험이 왜 더 이상 존재하지 않는지 기록하는 주석으로 대체.

---

## 5. 검증과 후속

### 통과

- GB10에서 `cargo test --profile test-fast --features cuda --lib server::streaming`: 25개 통과, 0개 실패.
- GB10에서 `cargo test --profile test-fast --features cuda --lib server::routes`: 116개 통과, 0개 실패.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`.
- `cargo fmt --all -- --check`.

### 덮지 않은 것

- 종단 SSE 프레임 캡처는 없다. 이번 변경이 쓰지 말자고 논증하는 바로 그 테스트이고, 내보내는 프레임에 바뀐 것도 없다. 같은 `KeepAlive` 값이 같은 `Sse`에 같은 순서로 닿는다.
- 수용 기준은 `--features metal,accelerate`를 지목한다. 이 장비는 Linux CUDA 박스라 그 조합을 빌드할 수 없고, 해당 코드에는 어떤 피처 게이트도 닿지 않는다.
- 기준의 `server::` 선택자는 필터 없이 돌리지 않고 실제로 채워진 두 절반으로 나눠 돌렸다. 필터 없는 선택자는 수백 개 테스트에 걸리고, 이 CUDA 호스트에서는 전체 lib 스위트가 이번 변경과 무관한 이유로 병렬 실행 중 중단된다.

### 후속

- #1133과 이번 변경이 들어가면서 keepalive는 간격 하나, 가드 하나, 부착 지점 하나가 되었다. 앞으로 추가되는 표면은 `sse_response`를 부르는 것으로 셋을 모두 얻고, 부르지 않는 표면은 조용히 빠지는 대신 이 배치 바깥에 있다는 사실이 눈에 띈다.
- 이 크레이트 안에서는 불변식이 구성으로 강제되지만, `axum::response::sse::Sse`는 외부 크레이트의 공개 타입이라 미래의 라우트가 직접 import하는 것을 기계적으로 막지는 못한다. `clippy.toml`에 이 타입을 `disallowed-types`로 넣고 `sse_response`를 가리키게 하면 주장이 리뷰어 검사에서 컴파일러 검사로 바뀐다. 이번에는 하지 않았다. 저장소에 `clippy.toml`이 없고, 워크스페이스 전역 린트 설정을 도입하는 것은 이 이슈가 요청한 범위보다 큰 결정이다. 리뷰에서 제기됐고 따로 볼 값어치가 있다.
