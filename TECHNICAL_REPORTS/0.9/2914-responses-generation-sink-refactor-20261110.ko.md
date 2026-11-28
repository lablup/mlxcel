# 기술 보고서: PR #2914 - refactor(server): Responses 생성을 HTTP 연결에서 분리

**날짜**: 2026-11-10
**상태**: 리뷰 중 (draft PR, #2896 종료, 에픽 #2903의 일부)
**언어**: Rust
**위험도**: 중간

---

## 요약

PR #2914는 스트리밍 `/v1/responses` 생성 로직을 HTTP 핸들러 `stream_create_response`에서 새 모듈 `src/server/routes/responses_run.rs`로 옮겨, SSE 연결 없이도 같은 생성을 실행할 수 있게 합니다. 백그라운드 모드(에픽 #2903)의 첫 단계입니다. `prepare_response_run`은 블로킹 작업에 필요한 모든 것을 소유하는 `PreparedRun`을 만듭니다. `drive_response_run`은 기존 `spawn_blocking` 클로저 본문을 그대로 옮긴 것으로, 모든 이벤트와 생명주기 단계를 새 `ResponseEventSink` 트레이트로 전달합니다. 기존 포그라운드 동작은 `SseResponseSink`가 그대로 재현합니다. 라우트, 플래그, 와이어 포맷은 바뀌지 않았습니다. 기존 라우트 테스트는 수정 없이 통과합니다(`server::routes` 552개 통과). 실제 `qwen3-0.6b-4bit` 체크포인트로 실행해 보니 스트리밍, 비스트리밍, API 취소, 클라이언트 연결 끊김이 모두 이전과 같이 동작했습니다. 비SSE 경로를 호출하는 코드는 아직 없으며, 첫 사용처는 #2898입니다.

---

## 1. 문제 정의

### 1.1 배경

백그라운드 모드(`background: true`)는 응답을 만든 즉시 반환하고, 클라이언트 없이 생성을 계속해야 합니다. 이 PR 이전의 유일한 스트리밍 경로는 네 군데에서 SSE 채널에 묶여 있었습니다.

- **이벤트 전달:** 생성 클로저가 `ResponseStreamSender::send_event`를 직접 호출했습니다.
- **취소:** 채널이 닫히면 요청의 취소 토큰이 켜졌습니다. 즉 읽는 쪽이 떠나는 것이 곧 취소였습니다.
- **저장:** `persist_response`와 조기 취소 시의 `store.insert`를 인라인으로 호출했습니다.
- **최종 출력:** 최종 객체의 `output`을 만드는 완료 항목 기록(`done_items`, #2396)이 sender 안에 있었습니다.

### 1.2 기존 문제

- **재사용할 수 있는 드라이버가 없었습니다.** 약 600줄의 엔벨로프 순서 처리(reasoning, message, tool-call 항목, compaction, 취소 처리)가 HTTP 핸들러의 클로저 하나에 들어 있었습니다. 백그라운드 모드를 만들려면 이를 복사해야 했을 것입니다.
- **출력이 전송 방식에 묶여 있었습니다.** `streamed_output()`이 SSE sender에 있었기 때문에, 최종 응답의 `output`이 어떤 전송 방식으로 생성했는지에 따라 달라졌습니다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 포그라운드와 백그라운드에 드라이버가 중복되어 서로 어긋남 | 높음 | 이 리팩터 없이는 높음 |
| 리팩터가 포그라운드 SSE 바이트나 취소 의미를 바꿈 | 높음 | 낮음 (그대로 이동, 테스트 무수정, 실제 체크포인트 검증) |

---

## 2. 변경 요약

| 파일 | 변경 |
|------|------|
| `src/server/routes/responses_run.rs` (신규) | `ResponseEventSink`, `PreparedRun`(+ `arm_admission`), `prepare_response_run`, `drive_response_run`, 비공개 `RunEvents` 래퍼, `SseResponseSink`, 옮겨 온 `streamed_output_text` |
| `src/server/routes/responses_run_tests.rs` (신규) | scripted provider 위에서 기록용 sink로 실행하는 드라이버 테스트 5개 |
| `src/server/routes/responses.rs` | `stream_create_response`를 prepare, 채널·sink 생성, spawn, admission 대기, `sse_response`만 남기도록 축소. `persist_response`는 `pub(crate)`, `count_reasoning_tokens`는 `pub(super)`로 변경 |
| `src/server/streaming_responses.rs` | `ResponseStreamSender`에서 `done_items`와 `streamed_output()` 제거 |
| `src/server/routes/responses_auto_compact.rs` | `send_compaction_events`가 `&dyn ResponseEventSink`를 받도록 변경 |

변경 후 호출 흐름:

```
create_response (검증, compaction)
  └─ stream_create_response
       ├─ prepare_response_run      → PreparedRun (렌더, 옵션, 캐시 컨텍스트, 큐 슬롯)
       ├─ responses_sse_channel_resumable → sender, stream, cancelled
       ├─ SseResponseSink::new      (/cancel용으로 `cancelled` 등록)
       ├─ run.arm_admission(cancelled)
       ├─ spawn_blocking(drive_response_run(run, cancelled, sink))
       └─ admission.wait → sse_response(stream)
```

---

## 3. 기술적 선택과 그 이유

### 3.1 부수 효과는 sink가, 응답은 드라이버가 맡음

**상황:** 형제 이슈 #2898이 병렬로 이 트레이트에 맞춰 작성되고 있었기 때문에, 이슈가 트레이트의 메서드 다섯 개를 고정해 두었습니다.

**결정:** 드라이버는 상태를 결정하고 모든 이벤트와 최종 객체를 만들며, 저장소나 채널은 건드리지 않습니다. 전달, 저장, 등록이 무엇을 뜻하는지는 sink가 정합니다. 완료 항목 기록은 드라이버의 비공개 `RunEvents` 래퍼가 맡습니다. 이 래퍼는 같은 트레이트를 구현하고 호출자의 sink로 전달합니다. compaction 이벤트도 이 래퍼를 거치므로 compaction 항목이 `streamed_output()`에 남고, `prepend_compaction_item`의 중복 제거도 그대로 동작합니다.

**이유:** 최종 객체의 `output`은 어느 sink가 실행했든 같아야 합니다. 기록을 드라이버에 두면 이 성질이 각 sink가 지켜야 하는 관례가 아니라 구조로 보장됩니다.

### 3.2 조기 취소도 `record_terminal`을 거침

**상황:** 첫 토큰 전에 확인된 취소는 원래 `store.insert`를 직접 호출했고, `persist_response`가 하는 대화(conversation) 추가는 하지 않았습니다. 트레이트에는 `record_terminal(&ResponseObject)` 하나뿐이고, 이 경우를 구분하는 플래그가 없습니다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 드라이버에 직접 store insert를 유지 | 추론이 필요 없음 | 백그라운드 sink가 맡아야 할 저장소를 드라이버가 건드림 |
| 트레이트에 메서드나 인자를 추가 | 명시적 | 형제 유닛이 기준으로 삼은 시그니처에서 벗어남 |
| **선택:** `cancelled` 상태이고 `output`이 빈 응답으로 `record_terminal`을 호출하고, sink는 정확히 이 형태일 때만 대화 추가를 생략 | 시그니처 유지, 모든 sink에 같은 규칙 | 불변식에 의존 |

**의존하는 불변식:** `build_response_object`는 function call을 내보내지 않으면 항상 message 항목을 추가합니다(`responses_translator.rs`의 "Message item" 블록). 따라서 그 밖의 모든 종료 응답에는 항목이 최소 하나 있습니다. 이 규칙은 트레이트 문서에 적혀 있어 #2898도 같은 조건을 적용합니다.

### 3.3 오류 경로는 `record_terminal`을 호출하지 않음

이전 동작을 그대로 유지합니다. 생성이 실패하면 `response.error`만 내보냅니다. 트레이트 문서에는 이벤트를 버리는 sink(예정된 백그라운드 sink)가 코드와 메시지를 알려면 `emit`에서 `Error` 이벤트를 읽어야 한다고 적었습니다. `finish_registration`은 모든 경로에서 한 번씩 실행됩니다.

### 3.4 `SseResponseSink`는 sender를 두 개 보유

`generation_finished`는 `&self`로 호출되므로, 잠금 없이는 sender를 교체할 수 없습니다. 그래서 sink는 `sender.after_generation()`을 미리 만들어 두고 `AtomicBool`로 전환합니다. 두 sender는 같은 `tx`의 복제본이므로, SSE 스트림은 드라이버가 sink를 버리는 시점에 정확히 끝납니다.

---

## 4. 검증

- **정적 검사:** `cargo fmt --check`, clippy(`-p mlxcel --lib --tests --features cuda -D warnings`), 환경 변수 문서 검사, 라이선스 헤더 검사가 모두 통과했습니다.
- **단위 테스트:** GB10에서 `--test-threads=1`로 실행한 결과 `server::routes` 552개 통과(1개 무시), `server::streaming_responses` 4개 통과, `server::reasoning_echo` 30개 통과입니다.
- **신규 테스트:**
  - 같은 토큰 스크립트에서 기록용 sink가 SSE 와이어와 같은 `(event, sequence_number)` 목록을 받습니다.
  - `record_terminal`은 한 번만 실행되며, `generation_finished`와 `finish_registration` 다음, 마지막 이벤트 전에 실행됩니다.
  - `on_started`는 한 번만 실행됩니다.
  - 오류 경로는 `response.error`만 내보냅니다.
  - 조기 취소는 `response.completed` 전에 빈 `cancelled` 응답을 하나 기록합니다.
- **실제 체크포인트(`qwen3-0.6b-4bit`):**
  - 스트리밍 요청은 sequence number가 0..66으로 연속이었고, `completed`로 끝났으며, 조회도 가능했습니다.
  - 비스트리밍 요청은 `completed`로 끝났습니다.
  - 스트림 도중 API로 취소하면 `status: cancelled`를 담은 `response.completed`로 끝났고, 다시 취소해도 결과가 같았습니다(멱등).
  - 클라이언트 연결을 끊으면 `cancelled (client disconnected)` 로그가 남았습니다.
- **리뷰:** 독립 정적 리뷰에서 CRITICAL과 HIGH는 없었습니다. MEDIUM 두 건은 반영했습니다. 종료 계약을 트레이트 문서에 명시했고, 빈 output 조건을 `cancelled`일 때로 좁혔습니다. 같은 리뷰에 따라 오류 경로 테스트와 조기 취소 테스트를 추가했습니다.

### 검증하지 않은 부분

- tool-call 스트리밍(#2396)에서 `streamed_output()` 경로를 비SSE로 확인하는 테스트는 없습니다. 이 경로는 여전히 SSE 라우트 테스트로만 확인됩니다.
- prefill 진행 콜백에서 `on_started`가 실행되는지는 테스트하지 않았습니다. scripted provider가 prefill 진행 콜백을 호출하지 않으므로, 토큰으로 시작되는 경우만 테스트했습니다.

---

## 5. 학습 포인트

- **먼저 옮기고, 그다음에 갈라냅니다.** 드라이버는 정규화한 텍스트 diff로 원래 클로저와 비교해 확인했습니다. 차이는 sink 호출과 지역 변수 이름 변경뿐입니다. 그래서 "바이트 호환"은 작성자의 주장이 아니라 리뷰어가 직접 확인할 수 있는 사실이 됩니다.
- **불변식은 모든 구현자가 읽는 곳에 둡니다.** 조기 취소 규칙은 원래 인라인 주석에만 있었습니다. 이를 트레이트 문서에 적어 두어야 두 번째 sink 구현(#2898)도 같은 규칙을 지킬 수 있습니다.
