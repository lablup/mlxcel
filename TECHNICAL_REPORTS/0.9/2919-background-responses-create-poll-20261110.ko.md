# PR #2919: 백그라운드 Responses 생성과 폴링

**날짜**: 2026-11-10
**상태**: 완료 (드래프트 PR로 열림; #2898을 닫으며 에픽 #2903의 일부)

## 요약

PR #2919는 단일 모델 서버에서 OpenAI Responses API의 백그라운드 모드를 켭니다. `background: true`인 `POST /v1/responses`는 이제 요청을 동기적으로 검증하고, `ResponsesStore`의 백그라운드 레지스트리(#2897)에 작업을 받아들인 뒤, 약 1 ms 안에 `queued` 객체와 함께 HTTP 200을 돌려줍니다. 이후 분리된 Tokio 태스크가 생성 전의 느린 작업을 수행하고, 생성은 새 `BackgroundSink`와 함께 블로킹 풀의 `drive_response_run`(#2896)에 넘깁니다. `GET /v1/responses/{id}`는 작업을 `queued`, `in_progress`, 종료 상태 순으로 폴링합니다. 작업 태스크나 드라이버의 패닉을 포함한 모든 종료 경로가 종료 객체를 정확히 하나 기록하므로 active 상태로 남는 작업이 없습니다. 포그라운드 응답의 직렬화 결과는 바이트 단위로 이전과 같습니다. 새 필드 `ResponseObject.background`는 값이 `true`일 때만 출력됩니다. 구현 리뷰와 보안 점검에서 CRITICAL이나 HIGH 문제는 없었고, 마무리 단계에서 LOW 항목 두 개를 고쳤습니다. 12개 파일, 약 +1350/-27이며 그중 약 560줄이 테스트입니다.

## 1. 문제 정의

이 PR 전에는 translator가 `background: true`를 400(`BackgroundUnsupported`)으로 거절했습니다. 두 선행 작업이 부품은 준비했지만 연결하지는 않았습니다.

- #2896은 Responses 생성을 HTTP 연결에서 분리했습니다. `prepare_response_run`이 `PreparedRun`을 만들고, `drive_response_run(run, cancelled, sink)`가 이를 임의의 `ResponseEventSink`로 구동합니다.
- #2897은 `ResponsesStore`에 수용 한도, 상태 전이, 원자적 종료 인계, `store: false`용 임시 엔트리를 갖춘 활성 작업 맵을 추가했습니다.

빠져 있던 것은 라우트 동작이었습니다. 어떤 검사를 동기로 남길지, 어떤 작업을 작업 안으로 옮길지, 상태를 `queued`에서 `in_progress`로 언제 바꿀지, 드라이버가 스트림 이벤트로만 알리는 실패를 어떻게 저장된 `failed` 객체로 만들지, 그리고 기존 GET, DELETE, `previous_response_id`, `conversation` 경로가 끝나지 않은 작업을 어떻게 다룰지가 정해져야 했습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/server/routes/responses_background.rs` (신규) | `BackgroundJob`, `validate_background_request`, `create_background_response`(503/429/400 매핑), `spawn_background_job`과 `spawn_background_job_with`(테스트용으로 드라이버 주입 가능), 작업 태스크 `run_background_job`, `run_prelude` |
| `src/server/routes/responses_background_sink.rs` (신규) | `JobRecorder`(최초 종료 기록, 실패 코드 매핑, `Drop` 가드)와 `BackgroundSink`(작업용 `ResponseEventSink`) |
| `src/server/routes/responses.rs` | `create_response`가 `background`로 분기하여 핸들러에서 비디오 확장과 자동 압축을 건너뜀; `retrieve_response`가 `background_snapshot`으로 폴백; `delete_response`가 활성 작업을 거절; `persist_response`에서 `append_conversation_turn` 분리 |
| `src/server/responses_translator.rs` | `BackgroundUnsupported` 제거; `previous_response_id`가 활성 작업을 가리키면 `PreviousNotFinished { id, status }`; `build_response_object`가 `background` 설정 |
| `src/server/responses_store.rs` | 압축 후 활성 작업의 입력 항목을 교체하는 `set_background_input_items` |
| `src/server/types/responses_response.rs` | `skip_serializing_if = "Option::is_none"`이 붙은 `background: Option<bool>` |
| `src/server/routes/responses_background_route_tests.rs` (신규) | 스크립트 프로바이더 기반 테스트 12개: 이슈의 수용 기준 전부, in-flight 레지스트리를 통한 취소, 종료 기록 후 드라이버 패닉 |
| `docs/responses-api.md` | "Background mode" 절 신설, 필드와 엔드포인트 표, 오류 케이스, 메모리 전용이라 종료 시 사라지는 동작 |

## 3. 기술적 선택과 그 이유

### 3.1 동기로 남기는 검사

요청만 들여다보는 검사는 모두 작업이 생기기 전에 400으로 남습니다. `context_management` 파싱, 번역(새 미완료 검사 포함), 미디어 지원 여부 검사, 실효 입력 존재, 샘플링 파라미터 범위, 도구 형태, 추론 예산, 구조화 출력 제약이 여기에 속합니다. 백그라운드 전용 거절(`stream: true`, 스토어 비활성, 작업 한도 0, 엔트리나 바이트 예산 0)은 `validate_background_request`에서 가장 먼저 수행됩니다. 구조화 제약을 create 시점에 만드는 것은 의도된 선택입니다. 잘못된 JSON 스키마는 호출자가 즉시 봐야 하는 클라이언트 오류이지, 폴링해서 찾아야 하는 `failed` 작업이 아닙니다.

수 초가 걸릴 수 있는 세 단계는 작업 안으로 옮겼습니다. `expand_request_video_parts`(프레임 디코딩), `apply_auto_compaction`(생성 패스 한 번 전체), `prepare_response_run`(미디어 읽기와 템플릿 렌더링)입니다. 이 단계의 오류는 `failed` 객체가 됩니다.

### 3.2 큐 검사는 탐침

`create_background_response`는 `can_accept_request()`와 `reserve_single_stream_queue_slot()`을 호출해 어느 쪽이든 실패하면 포그라운드와 같은 503을 돌려주고, 예약은 곧바로 해제합니다. 작업은 prelude가 끝난 뒤 `prepare_response_run` 안에서 자기 슬롯을 예약하는데, 이는 포그라운드가 슬롯을 예약하는 지점과 같습니다. prelude 동안 슬롯을 쥐고 있으면 큐 깊이 1에서 압축이 교착됩니다. 압축 패스도 자기 슬롯이 필요하기 때문입니다. 대가는 create와 작업 시작 사이에 큐가 차면 작업이 기다리지 않고 `server_overloaded`로 `failed`가 된다는 점입니다. 보안 리뷰는 `--responses-background-max-jobs`(64)가 슬롯 수와 큐 깊이와 무관하므로 큐보다 큰 버스트에서 일부 작업이 실패한다는 점을 MEDIUM으로 지적했습니다. 문서에 적었고, 슬롯을 기다리는 작업은 후속 과제로 남깁니다.

### 3.3 응답 전에 등록되는 단일 취소 토큰

`admit_background`는 `InFlightToken`을 돌려줍니다. `spawn_background_job_with`는 create가 응답하기 전에 바로 그 `Arc`를 스토어의 in-flight 레지스트리에 응답 id로 넣고, 작업은 같은 토큰을 `arm_admission`과 `drive_response_run`에 넘깁니다. 그래서 기존 `POST /v1/responses/{id}/cancel`(`cancel_in_flight`)이 변경 없이 생성까지 닿고, create 직후에 보낸 취소도 작업을 놓치지 않습니다. 클라이언트 연결을 참조하는 것이 없으므로 create 응답이나 폴링 연결을 끊어도 아무것도 취소되지 않으며, 테스트가 이를 확인합니다.

작업이 prelude 중일 때 도착한 취소는 prelude가 끝난 뒤 확인됩니다. 이때 작업은 빈 출력의 `cancelled`를 기록하고 디스패치하지 않습니다. 진행 중인 압축 패스는 중단되지 않습니다(압축 API가 토큰을 받지 않음). 취소는 압축이 끝난 뒤 적용됩니다.

### 3.4 `queued`에서 `in_progress`로

프로바이더는 `AdmissionWaiter`에 디스패치 시점에 응답하는데, 미디어 요청이라면 슬롯이 시작되기 전입니다. 따라서 waiter로는 "시작됨"을 알 수 없습니다. 상태는 드라이버가 첫 prefill 진행 콜백이나 토큰 콜백에서 호출하는 `BackgroundSink::on_started`에서 바뀝니다. 그래도 waiter는 프로바이더가 응답할 때까지 쥐고 있습니다. 응답 없이 버리면 생성이 취소되기 때문이며, 그 결과는 드라이버가 종료 없이 돌아왔을 때의 대체 오류로 쓰입니다.

### 3.5 어느 경로로 끝나든 종료는 하나

모든 종료 기록은 첫 호출이 이기는 `ResponsesStore::finish_background`를 거칩니다. 기록 출처는 다음과 같습니다.

- 드라이버 자신의 최종 응답이 `record_terminal`로 들어옴: `completed`, `incomplete`(`final_status` 판단), 또는 생성 중 `cancelled`.
- 생성 오류. 드라이버는 이 경로에서 `record_terminal`을 호출하지 않고 `ResponseStreamEvent::Error`로만 알립니다. `BackgroundSink::emit`이 이 이벤트를 가로채 이벤트의 코드와 메시지로 `failed`를 기록합니다. 워커가 종료된 경우도 `ChatWorkerGoneError`로 이 경로를 탑니다.
- prelude와 `prepare_response_run` 실패는 `JobRecorder::fail_with`를 거칩니다. 포그라운드의 오류 응답을 `error.code`로 바꾸는데, 4xx는 `invalid_prompt`, `server_error`가 아닌 503은 `server_overloaded`, 나머지는 `server_error`입니다.
- 블로킹 드라이버의 `JoinError`(패닉)는 작업이 아직 active일 때만 `server_error`로 기록합니다. 그래서 종료 기록 후의 패닉은 결과를 건드리지 않습니다(마무리 단계 수정).
- 작업이 active인 채 마지막 `JobRecorder` 핸들이 drop될 때의 `Drop`: 비동기 작업 태스크의 패닉이나 런타임 종료.

`JobRecorder::fail`은 먼저 `finish_in_flight`를 호출합니다. 클라이언트에게 이미 취소를 확인해 주었다면 `failed` 대신 `cancelled`를 기록하므로, 클라이언트가 "cancelled"를 들은 뒤 "failed"를 보는 일은 없습니다.

### 3.6 폴링에는 부분 출력이 없음

스냅샷은 종료 전까지 `output: []`과 사용량 0을 유지합니다. 이슈가 제안한 기본값입니다. 부분 텍스트는 스트리밍(#2900)의 몫이며, 여기서 노출하려면 토큰마다 드라이버 상태를 스토어로 복사해야 합니다.

### 3.7 압축 후 저장되는 입력

`admit_background`는 요청의 정규 입력 항목을 저장하지만, 자동 압축은 그 뒤에 실행되어 이 항목을 다시 씁니다. 이때 종료 응답의 출력에는 압축 캡슐이 들어갑니다. 원래 항목으로 저장하면 이를 `previous_response_id`로 이어 받는 요청이 압축 전 기록과 캡슐을 모두 재생합니다. `set_background_input_items`는 항목을 교체하고 활성 바이트 수를 조정합니다. 압축은 입력을 줄이기만 하므로 바이트 예산은 다시 확인하지 않습니다.

### 3.8 대화와 저장 기록

`finish_background`가 이미 종료 객체를 저장하므로 `persist_response`를 부르면 두 번 저장됩니다. 그래서 `persist_response`에서 `append_conversation_turn`을 분리했고(포그라운드 동작은 그대로), sink는 종료 시점에 같은 `effective_store` 규칙으로 이것만 호출합니다. 빈 출력의 취소이거나, 스토어가 너무 큰 객체 대신 `failed` 객체를 넣은 경우에는 대화에 추가하지 않습니다.

### 3.9 기존 엔드포인트와의 상호작용

- `retrieve_response`는 `store.get`을 먼저 시도하고 다음으로 `background_snapshot`을 봅니다. 활성 작업과, 임시 TTL이 끝나기 전의 `store: false` 결과가 여기서 읽힙니다.
- `delete_response`는 활성 작업에 대해 400을 돌려주며 먼저 취소하라고 안내합니다. 작업을 지우면 생성이 관찰자 없이 계속 돌기 때문입니다.
- `previous_response_id`가 활성 작업을 가리키면 400 `previous response '<id>' has not finished (status: <status>)`를 돌려줍니다. translator는 `store.get` 전에 `background_status`를 확인하므로, 두 호출 사이에 끝난 작업은 저장된 응답으로 찾아집니다. 이 규칙은 포그라운드 요청에도 적용됩니다.
- `store: false` 결과는 체이닝할 수 없습니다(`get`이 임시 엔트리를 숨김). 포그라운드와 같습니다.

## 4. 검증

- `gpu-lock` 아래 `cargo test --profile test-fast --features cuda -p mlxcel --lib server::routes::responses -- --test-threads=1`: 마무리 커밋 후 79개 통과. 구현 커밋에서 `responses_store`(30), `responses_translator`(24), `server::types::responses`(18) 통과. 백그라운드 테스트는 반복 실행에서도 통과(8회, 리뷰에서 추가 3회).
- clippy(`-p mlxcel --lib --tests --features cuda -D warnings`), `cargo fmt --check`, 환경 변수 문서 검사, 라이선스 헤더 검사 통과.
- GB10 실제 체크포인트(`qwen3-0.6b-4bit`, `--responses-background-max-jobs 1`): create가 1.2 ms에 200 `queued`를 반환; 폴링은 `in_progress`를 거쳐 출력과 사용량이 있는 `completed`; `store: false` 작업은 폴링으로 완료되었고 체이닝되지 않음; 두 번째 작업은 429 `rate_limit_exceeded`; 활성 작업에 대한 `previous_response_id`와 `DELETE`는 400; `background`와 `stream` 동시 지정은 400; 포그라운드 create에는 `background` 키가 없음. 이 작은 모델에서는 create 50 ms 뒤의 첫 폴링이 이미 `in_progress`였고, `queued` 폴링은 스크립트 테스트가 고정합니다.

## 5. 알려진 한계와 후속 과제

- 취소 의미론은 #2899의 몫입니다. `request_background_cancel`은 취소 확인 표시 없이 토큰만 바꿉니다. #2899가 라우트를 이 함수로 바꾸면 첫 토큰 전에 생성을 멈춘 취소가 `failed`로 기록되므로, 그때 `JobRecorder::fail`이 설정된 토큰도 `cancelled`로 취급해야 합니다. 끝난 `store: false` 작업의 취소는 404이고, 활성 작업 취소는 아직 예전의 최소 응답을 돌려줍니다.
- 슬롯과 큐 깊이를 넘는 작업 버스트는 초과분이 `server_overloaded`로 실패합니다(3.2).
- 백그라운드 id는 포그라운드 저장 응답처럼 API 키 범위로 묶이지 않으며, 추측할 수 없는 UUIDv4 id가 유일한 보호입니다. 실패한 작업은 드라이버의 오류 문구를 스토어 TTL 동안 `error.message`에 보관합니다.
- 폴링마다 `jobs` 뮤텍스를 잡고 `inner` 쓰기 락 아래에서 만료 엔트리를 정리합니다(#2897 코드가 이제 자주 불리는 GET 경로에 있음).
- #2897에서 이어진 항목: `DELETE`가 `store: false` 종료 엔트리도 지우고, 중복 id 수용은 활성 작업을 대체합니다(id는 서버 UUID).
- 작업은 메모리에만 있습니다. 단일 모델 서버는 시그널 핸들러가 없어 프로세스가 끝나면 작업이 사라지며, 문서에 적었습니다.

## 6. 형제 이슈를 위한 인터페이스

- `ResponseObject.background: Option<bool>`은 `build_response_object`가 `request.background.filter(|b| *b)`로 설정합니다. 따라서 드라이버의 최종 응답과 스트림 이벤트 페이로드를 포함해 백그라운드 요청으로 만든 모든 객체에 `true`가 실립니다.
- #2900(스트림과 재개)은 `validate_background_request`의 `stream: true` 거절을 없애고, 지금은 `Error`만 가로채는 `BackgroundSink::emit`에서 이벤트를 팬아웃합니다.
- #2899(취소)는 `spawn_background_job_with`가 등록하는 `InFlightToken`, `finish_registration` 확인, `JobRecorder::fail`의 취소 규칙(5절)을 다룹니다.
- #2901(라우터 역할): 이 PR은 `router_server.rs`를 건드리지 않습니다. 작업은 태스크와 블로킹 스레드를 통해 `AppState`와 모델 프로바이더를 쥡니다.

## 7. 학습 포인트

- 드라이버가 일부 결과를 이벤트로만 알린다면, 결과를 저장하는 sink는 그 이벤트를 읽어야 합니다. `record_terminal`만 믿으면 실패한 작업이 영원히 active로 남습니다.
- 공유 recorder의 `Drop` 가드와 첫 기록이 이기는 스토어 메서드를 조합하면, 모든 await 지점을 감싸지 않고도 패닉을 포함해 작업당 종료 하나를 보장할 수 있습니다.
- 요청 핸들러의 일을 분리된 작업으로 옮기면 그 동안 어떤 자원을 쥘 수 있는지가 달라집니다. 자기 슬롯이 필요한 압축 패스 동안 큐 슬롯을 쥐면 깊이 1에서 교착이 생깁니다.
