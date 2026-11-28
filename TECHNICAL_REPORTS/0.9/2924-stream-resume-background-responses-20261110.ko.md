# PR #2924: 백그라운드 Responses 스트리밍과 재개

**날짜**: 2026-11-10
**상태**: 완료 (드래프트 PR로 열려 있음; #2900을 닫으며 에픽 #2903의 일부)

## 요약

PR #2924는 `POST /v1/responses`에서 `background: true`와 `stream: true`를 함께 쓸 수 있게 하고, 연결이 끊긴 클라이언트가 `GET /v1/responses/{id}?stream=true&starting_after=N`으로 다시 이어 받을 수 있게 합니다. OpenAI Responses 규약이 설명하는 동작입니다. 스트리밍 백그라운드 작업마다 `BackgroundEventLog`가 하나씩 붙습니다. 직렬화된 SSE 이벤트를 0부터 빈틈없이 번호 매겨 담는, 크기가 제한된 재생 가능 로그입니다. 생성 요청의 응답 스트림과 모든 재개 요청은 이 로그 위의 서로 독립된 리더입니다. 연결이 끊기면 그 리더만 끝나고 작업은 종료 상태까지 계속 실행됩니다. `qwen3-0.6b-4bit`(GB10, CUDA)에서 시퀀스 39에서 끊은 스트림이 40부터 재개되어 496의 `response.completed`까지 빈틈이나 중복 없이 이어졌고, OpenAI Python SDK(3.48.0)의 `responses.create(background=True, stream=True)`와 `responses.retrieve(id, stream=True, starting_after=29)`도 같은 결과를 냈습니다. 포그라운드 스트리밍은 바뀌지 않았습니다. 구현 리뷰에서는 CRITICAL이나 HIGH가 없었고, 보안 점검에서 HIGH 한 건(멈춘 리더가 축출된 로그를 메모리에 붙잡아 둠)을 찾아 더 작은 발견 세 건과 함께 고쳤습니다. 28개 파일, 약 +2418/-52이며 그중 약 1210줄이 테스트입니다.

## 1. 문제 정의

#2898 이후 백그라운드 작업은 모든 스트림 이벤트를 버리는 `BackgroundSink`로 실행되었고, `validate_background_request`는 `stream: true`를 400으로 거부했습니다. 진행 상황이 필요한 클라이언트는 폴링해야 했습니다. OpenAI 규약은 작업 이벤트를 `sequence_number`와 함께 스트리밍하고, 연결을 잃은 클라이언트가 마지막으로 본 번호 다음부터 재개하게 합니다.

서버에는 이미 재생 버퍼 `src/server/stream_session.rs`(#1444)가 있었지만, 키가 `(owner, conversation_id)`이고 커서가 바이트 오프셋입니다. 리더마다 `sequence_number`를 바이트 오프셋으로 변환하는 방식은 이슈에서 기각되었습니다. 필요한 것은 시퀀스 번호로 주소 지정되는 새 로그와, 명확한 수명(작업 실행 중 누가 소유하는지, 작업이 끝나면 어디로 가는지, 메모리가 언제 해제되는지), 그리고 여러 클라이언트가 생성 속도를 떨어뜨리지 않고 동시에 재생과 추종을 할 수 있는 리더 모델이었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/server/responses_background_events.rs` (신규) | `BackgroundEventLog`(바이트 예산이 있는 `VecDeque<(u64, &'static str, Arc<str>)>`), 커서 해석(`CursorError::Evicted` / `Ahead`), 제한된 읽기, 종료 이벤트 준비, 리더가 소비하는 SSE 스트림 `tail_events` |
| `src/server/routes/responses_background_stream.rs` (신규) | 스트리밍 생성 응답, `resume_stream`, `RetrieveResponseQuery`(`stream`, `starting_after`)와 400/404 매핑 |
| `src/server/responses_store.rs` | `background_event_log(id)`, `attach_background_event_log`, `background_event_log_dropped`; 작업 레코드와 종료 `Entry`의 로그; `finish_background`에서 종료 이벤트 추가; 로그 바이트를 `size_bytes`와 `approximate_total_bytes`에 반영; `background_event_log_bytes` 설정 |
| `src/server/routes/responses_run.rs` | `PreparedRun::background_opening: Option<u64>`: 값이 있으면 드라이버가 `response.created`를 건너뛰고 그 값부터 번호를 매기며 첫 진행 콜백에서 `response.in_progress`를 보냄 |
| `src/server/routes/responses_background.rs` | `stream: true` 400 제거; 승인 시 로그 생성과 `response.created` / `response.queued` 추가; 로그 예산 0이면 스트리밍 생성 거부 |
| `src/server/routes/responses_background_sink.rs` | `BackgroundSink::emit`이 이벤트를 로그로 보냄; 드라이버의 `Error` 이벤트는 여전히 `failed`로 기록 |
| `src/server/routes/responses.rs` | `retrieve_response`가 쿼리를 받아 `stream=true`이면 `resume_stream`으로 분기 |
| `src/server/types/responses_stream.rs` | `ResponseStreamEvent::Queued`(`response.queued`), `sequence_number_mut` |
| 플래그 연결 | 두 바이너리, `serve`, `cli_input`, `startup`에 `--responses-background-event-log-bytes` / `MLXCEL_RESPONSES_BACKGROUND_EVENT_LOG_BYTES`(기본 4 MiB) |
| 테스트 (신규) | 로그 단위 테스트, 스토어 이벤트 로그 테스트, 라우트 테스트 모듈 두 개(스트림 동작과 오류 사례), 백그라운드 압축 스트림 테스트 하나 |
| 문서 | `docs/responses-api.md`의 "Streaming and resuming" 절과 플래그 행; `docs/environment-variables.md` |

## 3. 기술적 선택과 그 이유

### 3.1 번호는 쓰는 쪽이 아니라 로그가 매긴다

하나의 로그에 세 곳이 이벤트를 씁니다. 승인 경로(`response.created`, `response.queued`), 생성 드라이버(그 사이의 모든 이벤트), 스토어(종료 이벤트)입니다. 각자 번호를 매길 수도 있지만, 그러면 둘의 판단이 어긋날 때마다(예를 들어 드라이버가 실행되기 전에 프렐류드 실패로 작업이 끝날 때) 빈틈이나 중복이 생깁니다. 그래서 로그가 `append`에서 `sequence_number`를 직접 부여하고, 번호가 다른 이벤트는 다시 찍습니다. 드라이버에도 시작 번호(`background_opening`)를 알려 주므로 자체 이미터가 대개 일치하고, 다시 찍는 것은 예외적인 경우입니다.

### 3.2 승인 시점의 시작 이벤트

이슈는 순서 변경을 `PreparedRun` 필드에 두었습니다. PR은 그 필드를 유지하되 `response.created`(상태 `queued`)와 `response.queued`를 프렐류드 이전, 작업 승인 시점에 추가합니다. 그렇지 않으면 스트리밍 생성 요청은 비디오 디코딩이나 자동 압축이 끝날 때까지(수 초가 걸릴 수 있음) 응답 id조차 받지 못합니다. 드라이버는 `background_opening: Some(2)`를 보면 자체 `response.created`를 건너뛰고 2부터 번호를 매기며, 첫 진행 콜백에서 `response.in_progress`를 보냅니다. 이 시점은 `on_started`가 저장된 상태를 바꾸는 시점과 같습니다. `None`이면 드라이버 출력은 이전과 바이트 단위로 같고, 포그라운드 스트림은 항상 `None`입니다.

### 3.3 종료 이벤트는 스토어에서 한 번

백그라운드 작업이 끝나는 경로는 많습니다. 정상 드라이버 결과, `failed`로 변환되는 드라이버 `Error` 이벤트, `JobRecorder::fail`, `cancel_before_start`, `JobRecorder::drop`, 드라이버 `JoinError`, 그리고 너무 큰 결과를 스토어가 `failed` 객체로 대체하는 경우입니다. 각 경로에서 종료 이벤트를 추가하면 실수하기 쉽고 폴링 결과와 어긋날 수 있습니다. PR은 이미 첫 호출만 반영되는 `ResponsesStore::finish_background` 안에서, 실제로 저장되는 객체로부터 종료 이벤트를 추가합니다. 따라서 #2899의 취소 작업이 나중에 추가할 경로를 포함한 모든 종료 경로가, 폴링한 객체와 일치하는 종료 이벤트(`response.completed`, `response.incomplete`, `response.failed`) 정확히 하나로 스트림을 끝냅니다. 취소된 작업은 #2870 이후의 포그라운드처럼 `status: "cancelled"`를 담은 `response.completed`로 끝납니다.

스토어의 잠금 규율을 지키기 위해 종료 이벤트는 `finish_background`가 jobs 잠금을 잡기 전에 직렬화됩니다(`prepare_terminal`은 다음 시퀀스 번호만 읽습니다). 잠금 안에서는 번호가 여전히 맞고 저장 객체가 대체되지 않았을 때 준비된 이벤트를 그대로 넣고, 아니면 로그 잠금 아래에서 다시 직렬화합니다. 처음 버전은 jobs 뮤텍스 아래에서 직렬화했고, 보안 점검에서 옮겼습니다.

### 3.4 예산, 축출, 수명

로그는 새 이벤트가 `--responses-background-event-log-bytes`에 들어가지 않으면 가장 오래된 이벤트를 버리지만, 가장 최근 이벤트는 항상 남기므로 아주 작은 예산에서도 종료 이벤트가 살아남습니다. 대가로 로그는 예산을 이벤트 하나만큼 넘을 수 있습니다. 작업이 실행되는 동안에는 작업 레코드가 로그를 소유하고, `finish_background`가 로그를 종료 `Entry`로 옮겨 그 바이트를 `size_bytes`에 더하며, 스토어 축출이 로그를 해제합니다. 응답과 함께 `--responses-store-max-bytes`에 들어가지 않는 로그는 응답을 축출하는 대신 종료 시점에 버리고, 엔트리는 로그를 버렸다는 사실을 기억해 나중의 재개 요청에 정확한 메시지를 줍니다. 실행 중인 작업의 로그는 `approximate_total_bytes`에는 포함되지만 승인 예산 검사에는 들어가지 않으므로, 실행 중인 스트리밍 작업들이 스토어 예산 밖에서 최대 `max_jobs` 곱하기 로그 예산(기본값에서 256 MiB)을 쓸 수 있습니다. 설정으로 한정되며 문서화되어 있습니다.

### 3.5 리더

리더는 커서 하나와 `Weak` 핸들 하나입니다. 읽을 때마다 로그 잠금을 잡고 최대 256개 이벤트 또는 64 KiB(최소 한 이벤트)를 복사한 뒤 잠금을 풀고 프레임을 내보냅니다. 따라잡았으면 다시 읽기 전에 `Notified` 퓨처를 먼저 만들어, 읽기와 await 사이에 들어온 추가도 리더를 깨우게 합니다. 1초 주기 깨우기는 안전장치입니다. 추가는 잠금을 푼 뒤 `notify_waiters`를 부르고 리더를 기다리지 않으므로, 느린 클라이언트가 생성을 늦출 수 없습니다. 유지 구간 밖으로 밀려난 리더는 `code: "stream_events_evicted"`와 재개 400과 같은 메시지를 담은 `response.error` 이벤트 하나를 받고 끝납니다. 종료 이벤트를 보낸 리더는 더 이상 로그가 필요 없습니다.

보안 점검은 첫 버전이 스트림 전체 동안 강한 `Arc`를 쥐고 있어서, 스트림을 열고 읽기를 멈춘 클라이언트가 스토어가 엔트리를 축출한 뒤에도 최대 4 MiB를 붙잡을 수 있었음을 찾았습니다. 연결 수 외에는 제한이 없었습니다. `Weak` 핸들이 이를 고칩니다. 리더는 작업이 어차피 로그를 소유하는 라이브 추종 대기 중에만 핸들을 승격하고, 종료 이벤트 전에 로그가 사라지면 `response.error` 이벤트를 받고 닫힙니다.

### 3.6 재개 커서 규칙

`starting_after`는 `u64`로 파싱되며 그 밖의 값은 400입니다. 커서가 있으면 더 큰 번호의 이벤트를 재생한 뒤 추종합니다. 커서가 없으면 버린 이벤트가 없을 때만 0부터 재생합니다. 유지된 가장 오래된 번호에서 1을 뺀 값보다 작은 커서는 유지된 가장 오래된 `sequence_number`를 알려 주고 응답을 GET하라고 안내하는 400입니다. 부여된 가장 최근 번호를 넘는 커서는 그 번호를 알려 주는 400입니다(보안 점검에서 추가; 첫 버전은 끝난 작업에서 종료 이벤트 없는 빈 스트림을 돌려줬습니다). 가장 최근 번호 자체는 유효한 커서로, 실행 중인 작업에서는 추종하고 끝난 작업에서는 바로 닫힙니다. 로그가 없는 응답은 `background: true`와 `stream: true`로 만들어지지 않았다는 400, 모르는 id는 404이며, `stream=true`가 없으면 핸들러는 이전과 똑같이 동작합니다. 커서가 재개 수단이므로 백그라운드 작업에서는 `X-Conversation-Id`를 무시합니다.

## 4. 검증

- `gpu-lock` 아래 `cargo test --profile test-fast --features cuda -p mlxcel --lib responses -- --test-threads=1`: 231개 통과. `cli_input`과 두 바이너리의 CLI 플래그 테스트도 통과.
- `cargo clippy -p mlxcel --lib --tests --profile test-fast --features cuda -- -D warnings`, `cargo fmt --all -- --check`, `scripts/ci/check_env_var_docs.py`, `scripts/insert_apache_header.py --check`: 문제 없음.
- 이슈의 수용 기준 13개 모두 테스트가 있습니다. 동시 리더 두 개(하나는 재생, 하나는 추종)가 같은 시퀀스를 받는 경우, 느린 리더의 밀려남, 작은 예산의 로그가 종료 이벤트를 지키는 경우, 스크립트된 일시 정지 동안의 keepalive 프레임, 축출 시 로그 바이트 해제 등이 포함됩니다.
- 최종 헤드(`0a0e49ba`)에서 실제 모델, `mlxcel-server -m models/qwen3-0.6b-4bit`, 출력 500 토큰: curl을 시퀀스 39에서 끊고 40부터 496의 `response.completed`까지 연속으로 재개; SDK는 29에서 끊고 30..498을 받아 `response.completed`로 끝남, 커서 없는 재생은 동일; `starting_after=abc`는 400, 모르는 id는 404, 포그라운드 id에 `stream=true`는 400.

## 5. 남은 위험과 후속 작업

- 실행 중인 작업의 로그는 승인 예산 밖에 있습니다(3.4절). 승인 시점에 로그 예산을 예약하는 후속 작업을 고려할 수 있습니다.
- 응답당 리더 수에 상한이 없습니다. `Weak` 수정 이후 리더 하나의 비용은 연결 하나와 최대 한 번의 읽기 배치입니다.
- `?stream=a&stream=b`처럼 중복된 쿼리 키는 이제 무시되지 않고 `Query` 추출기의 평문 400을 받습니다.
- `request_background_cancel`로 첫 토큰 전에 생성을 멈춘 취소는 여전히 `failed`로 기록되므로(#2899에 남겨진 알려진 공백) 그 스트림은 `response.failed`로 끝납니다. 스트림은 스토어가 기록한 대로 따라가므로 기록기를 고치면 스트림도 고쳐집니다.
- `admit_background`에서 응답 id가 충돌하면 이전 작업의 로그가 종료 이벤트 없이 버려집니다. id는 무작위이므로 사실상 일어나지 않습니다.
- 실행하지 않은 것: Metal, 라우터 풀(#2901), 전체 lib 테스트, CI.

## 6. 학습 포인트

- 여러 작성자가 순서가 있는 하나의 스트림을 공유하면, 순서는 스트림이 매기게 합니다. 그러면 어느 작성자가 실행되거나 실패하든 빈틈이 생기지 않습니다.
- 종료 이벤트는 종료 상태가 결정되는 곳에 둡니다. 첫 호출만 반영되는 스토어 전이 하나에서 내보내자 일곱 개의 종료 경로를 하나도 건드리지 않고 다룰 수 있었습니다.
- 재생 리더는 버퍼를 약하게 쥐어야 합니다. 연결마다 강한 핸들을 쥐면 읽기를 멈춘 클라이언트 앞에서 스토어 축출이 조용히 무력해집니다.
- `stream_session` 패턴은 그대로 옮겨 쓸 수 있습니다. await를 넘어 잡지 않는 `std::sync::Mutex`, 재확인 전에 만드는 `Notified`, 보험으로 두는 주기적 깨우기입니다.
