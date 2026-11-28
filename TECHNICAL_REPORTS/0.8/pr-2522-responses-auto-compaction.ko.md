# PR #2522: Responses 자동 압축(phase B)

**Date**: 2026-10-21
**Status**: Completed (PR open, #2397 해결)

## 요약

PR #2522는 #2397을 마무리한다. phase A(PR #2520)는 명시적인 `POST /v1/responses/compact`를 추가했다. 이 PR은 `/v1/responses`가 스스로 압축하게 만든다. 요청에 `context_management: [{"type":"compaction","compact_threshold":N}]`이 있으면, render한 기록이 N token에 이르는 순간 생성 전에 기록을 압축한다. 그러면 response output은 compaction item으로 시작한다. 스트리밍에서는 그 item 앞에 `mlx.compaction.started`와 `mlx.compaction.completed`를 보낸다. 이런 요청은 남은 context만큼만 생성하고, window를 채운 응답은 `incomplete`로 끝난다. `context_management`가 없는 요청은 바뀌지 않는다. 테스트가 스트리밍과 비스트리밍 경로 모두에서 이를 고정한다. 리뷰와 보안 점검에서 CRITICAL이나 HIGH는 없었고, MEDIUM 네 건을 고쳤다. 범위에서 뺀 항목은 #2523으로 등록했다. 변경 규모는 15개 파일, +1356/-23이며, 이 중 약 750줄이 테스트다.

## 1. 변경 요약

| 영역 | 변경 |
|---|---|
| `routes/responses_auto_compact.rs`(신규) | `context_management` 해석, 임계값 판정, 압축 호출, 출력 상한, `incomplete` 표시, reasoning budget 조정 |
| `routes/responses.rs` | 두 경로의 hook: token 계산, 압축, 상한 적용, 생성. 스트림 이벤트와 `output_index` 0의 capsule |
| `responses_compaction.rs` | `compact_history_counted`는 임계값 판정에 쓴 계산값을 재사용한다. `NoSplitPoint` 오류 추가 |
| `types/responses_stream.rs`, `types/responses_request.rs`, `responses_translator.rs` | 이벤트 variant 2개, 요청 필드, 필드가 있을 때만 펼친 item 보관 |
| 테스트 | `responses_auto_compact_tests.rs`, `responses_auto_compact_chain_tests.rs`. scripted provider가 `length`로 끝나는 경우를 지원 |
| 문서 | `docs/responses-api.md`, `docs/server-features.md`, `CHANGELOG.md` |

## 2. 기술적 선택과 그 이유

### 2.1 압축 시점

token 계산은 전체 기록을 대상으로 한다. chain으로 이어진 `previous_response_id` 또는 `conversation` item과 `input`을 합치고, capsule은 펼친 상태로 센다. 계산에는 phase A의 backend를 쓰며, 이는 `/v1/responses/input_tokens` 경로와 같다. 압축은 계산값이 `compact_threshold` 이상(`>=`)일 때 실행된다. upstream과 OpenAI의 "reaches"를 따른 것이고, 이슈 본문은 "exceeds"였다. 기록 때문에 출력할 공간이 남지 않으면 임계값 아래에서도 압축한다. phase A의 선택, capsule, 요약 코드는 바꾸지 않고 그대로 쓴다. 임계값 판정에 쓴 계산값을 재사용하므로 같은 계산을 반복하지 않는다. 자동 요약의 token budget은 upstream을 따라 context/16을 쓰고 128..1024로 제한한다.

### 2.2 출력 상한은 `context_management` 요청에만 적용한다

현재는 `length`로 끝난 모든 응답을 `completed`로 보고한다. `--context-shift`를 쓰면 생성이 window를 넘어서 계속될 수도 있다. 상한을 모든 요청에 적용하면 이 동작이 바뀐다. 그래서 상한은 이 필드가 있는 요청에만 적용하며, 실제로 압축했는지와는 상관없다. `max_tokens`는 `min(max_tokens, context - prompt_tokens)`가 되고, 뺄셈은 saturating 연산이다. 남은 공간이 없으면 `context_length_exceeded`를 반환한다. `length`로 끝난 응답은 다음 필드를 갖는다.
- `status: "incomplete"`
- `incomplete_details.reason: "max_output_tokens"` (OpenAI와 upstream의 값)
- 잘린 message item의 status는 `incomplete`
- `completed_at`은 null

스트림은 `response.incomplete`로 끝난다.

### 2.3 실패는 스트림을 열기 전에 처리한다

`ResponseStreamSender`는 생성 task에서 `blocking_send`로 보낸다. 그래서 압축은 SSE 응답이 시작되기 전에 실행된다. 실패하면 두 경로 모두 HTTP 오류를 받는다. admission 단계의 overflow도 이미 이렇게 처리된다. 형식이 잘못된 `context_management`는 어떤 작업도 하기 전에 400으로 거부한다. 배열이 아닐 때, 항목이 둘 이상일 때, 알 수 없는 type일 때, 임계값이 1..1e8 밖일 때가 여기에 해당한다. 분할할 지점이 없으면 기록이 아직 context에 들어가는 동안은 압축 없이 진행한다. upstream과 같은 동작이다. 들어가지 않으면 `context_length_exceeded`를 반환한다. 두 이벤트는 압축 전후의 token 수를 알려 줄 뿐, 진행 상황을 실시간으로 보내지는 않는다.

### 2.4 스트림 순서와 저장 형태

`response.in_progress` 뒤에 다음 순서로 보낸다.
1. `mlx.compaction.started`
2. `mlx.compaction.completed`
3. capsule의 `output_item.added`, `output_item.done` (index 0)

생성된 item은 index 1부터 시작한다. `prepend_compaction_item`은 서버가 만든 `cmp_` id로 item을 찾는다. 그래서 최종 객체를 만드는 두 경로 모두 capsule을 index 0에 정확히 한 번 넣는다. 기록된 item으로 객체를 다시 만드는 #2504의 tool-call 경로도 마찬가지다. 저장된 response는 instruction만 입력으로 갖고, capsule은 `output[0]`에 둔다. 다음 hop은 기존 output-to-input 변환으로 instruction, capsule, 이후 turn을 다시 만든다. 그 결과 요약이 재생되며, 임계값 아래에서는 다시 압축하지 않는다.

### 2.5 리뷰 지적 사항

| 심각도 | 지적 | 수정 |
|---|---|---|
| MEDIUM(리뷰) | `conversation` 경로의 자동 압축 테스트가 없었다 | `29dfec04` |
| MEDIUM(리뷰) | `context_management` 없는 요청이 바뀌지 않는다는 스트리밍 테스트가 없었다 | `29dfec04`: 이벤트 순서와 최종 객체를 정확히 확인 |
| MEDIUM(보안) | reasoning budget을 상한 적용 전의 값 기준으로 정해서, 상한이 걸린 응답이 reasoning에 budget을 모두 쓸 수 있었다 | `b078c984`: budget을 상한 적용 후 `max_tokens`보다 1 작게 제한 |
| MEDIUM(보안) | token 계산이 slot admission보다 먼저 실행되어, 바쁜 서버도 계산 작업을 했다 | `b078c984`: `can_accept_request()`를 먼저 실행하고 기존 503 반환 |
| LOW(리뷰) | 상한이 걸린 응답에서 item status가 `completed`로 남고 `completed_at`도 설정되었다 | `288f9eec` |

## 3. 검증

| 항목 | 결과 |
|---|---|
| `responses_auto_compact` | 15개 통과 |
| `responses_compact` | 28개 통과 |
| `server::routes::` | 508개 통과, 1개 ignored |
| `server::responses`, `server::types`, `streaming_responses` | 각각 74, 107, 4개 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | 문제 없음 |

모든 실행은 macOS에서 `MLXCEL_BUILD_METAL=0 --profile test-fast --lib`로 했다. 새로 추가한 busy server 테스트는 새 gate가 없어도 이 harness에서 통과한다. 뒤 단계가 이미 503을 반환하고, 텍스트 token 계산은 관찰할 수 있는 흔적을 남기지 않기 때문이다. gate가 동작하는지는 코드를 읽어 확인했다.

## 4. 미검증 항목과 남은 항목

- **미실행:** `mlx-community/Qwen3-30B-A3B-4bit`와 OpenAI SDK를 쓰는 실제 모델 통합 검증. 이 호스트에는 Metal도 CUDA도 없다. `context_management`와 이미지·비디오 prompt를 함께 쓰는 경우도 테스트가 없다.
- **알려진 한계(보안 MEDIUM, 고치지 않음):** 압축이 일어나는 스트리밍 요청은 요약 생성을 요청 task에서 실행한다. 클라이언트가 연결을 끊어도 요약 생성은 취소되지 않는다. 비스트리밍 요청과 `/compact`도 이미 같은 방식으로 동작하며, slot admission이 비용 상한을 둔다.
- **LOW:**
  - `/v1/responses/input_tokens`는 `context_management`를 검증하지 않고 무시한다.
  - 분할 지점이 없으면 이벤트를 보내지 않는다. upstream은 `started`만 보낸다.
  - 자동 요약이 실패하면 압축 없이 생성하지 않고 요청 전체가 실패한다.
  - token 계산용 요청을 만들 때 요청 전체를 한 번 복제한다.
- **후속 이슈 #2523:** Chat Completions와 `/v1/messages` 자동 압축, Codex `compaction_trigger`, 여러 단계에 걸친 요약.
- **규모:** +1356줄로 1,000줄 가이드라인을 넘는다. 테스트를 뺀 코드는 약 600줄이다.

## 5. 학습 포인트

- 스트림 sender가 executor를 막으면 안 될 때는, 실패할 수 있는 작업을 스트림을 열기 전에 끝내야 한다. 그래야 실패해도 반쯤 열린 스트림이 아니라 일반 HTTP 오류로 응답한다.
- opt-in 필드 뒤에서만 동작이 바뀐다면, 그 필드가 없는 경로가 바이트 단위로 같다는 테스트를 모든 전송 방식에 두어야 한다. 그렇지 않으면 opt-in이 어느새 기본 동작이 된다.
