# PR #2925: 백그라운드 cancel이 전체 Response를 반환하도록 수정

**날짜**: 2026-11-10
**상태**: 완료 (PR 열림; #2899를 닫으며 에픽 #2903의 일부)

## 요약

PR #2925는 `background: true` 응답에 대한 `POST /v1/responses/{id}/cancel`이 작업의 최종 Response 객체를 반환하게 합니다. 이 객체는 뒤이어 호출한 `GET /v1/responses/{id}`의 결과와 바이트 단위로 같습니다. 이전에는 포그라운드 스트림용 세 필드 envelope을 돌려주었고, 이미 끝난 `store: false` 작업에는 404를 돌려주었으며, 종료 객체가 기록되기 직전의 짧은 구간에는 존재하는 작업에도 404를 낼 수 있었습니다. 이제 cancel은 멱등이고 응답은 항상 기록된 종료 객체이므로 `client.responses.cancel()`이 기대하는 동작과 맞습니다. 포그라운드 cancel은 그대로입니다. 보안 리뷰에서 HIGH 문제 하나(취소된 queued 작업이 작업 한도 밖에서 준비 작업을 계속 실행함)가 나왔고 두 번째 커밋에서 고쳤습니다. 9개 파일, +968/-36이며 그중 약 600줄이 테스트입니다.

## 1. 문제 정의

PR #2919가 백그라운드 모드를 켰지만 cancel은 같은 프로세스에서 열려 있는 스트림을 전제로 만든 기존 포그라운드 라우트에 남아 있었습니다.

- 응답이 Response 객체가 아니라 포그라운드 envelope(세 필드)여서 OpenAI 클라이언트가 파싱할 수 없었습니다.
- 끝난 `store: false` 작업의 cancel이 404였습니다. 보존 기간 안의 GET은 같은 작업을 돌려줍니다.
- 작업의 in-flight 등록이 사라진 뒤 종료 객체가 기록되기 전까지, 작업은 포그라운드 라우트가 보는 두 곳 어디에도 없었습니다. 그래서 존재하는 작업에 404가 날 수 있었습니다.
- 첫 토큰 전에 생성을 멈추는 cancel은 드라이버 오류로 나타났고 `JobRecorder::fail`이 `cancelled`가 아니라 `failed`로 기록했습니다(#2919 리포트 5절에 적은 리뷰어 MEDIUM).

이슈가 요구하는 계약은 cancel이 멱등이고, 최종 Response를 돌려주며, GET과 절대 어긋나지 않는 것입니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/server/routes/responses.rs` | `cancel_response`가 먼저 `cancel_background_response`를 호출하고, `None`이면 바뀌지 않은 포그라운드 코드로 넘어감 (13줄) |
| `src/server/routes/responses_background.rs` | `BACKGROUND_CANCEL_WAIT`(5초), `cancel_background_response`, 작업의 생성 전 작업과 종료 기록을 경주시키는 `unless_cancelled`, 모듈 문서 |
| `src/server/responses_store_cancel.rs` (신규) | `cancel_background`, `background_cancel_requested`, `force_background_cancel`, `record_cancelled_locked` |
| `src/server/responses_store.rs` | 작업의 `cancel_requested` 플래그, cancel 경로가 종료 기록을 공유하도록 `finish_background`에서 분리한 `record_terminal_locked`, 플래그를 설정하고 토큰을 `Release`로 저장하는 `request_background_cancel`, 락 순서 문서 |
| `src/server/routes/responses_background_sink.rs` | `JobRecorder::terminal_recorded`, cancel 요청이 있으면 `cancelled`로 기록하고 원래 오류를 debug 로그로 남기는 `fail`, cancel 등록을 정리하는 `Drop`, 라우트가 종료 객체를 먼저 기록했을 때 대화 추가를 건너뜀 |
| `src/server/routes/responses_background_cancel_tests.rs` (신규) | 테스트 10개. `responses_background_route_tests.rs`의 자식 모듈이며 그 파일이 500줄을 넘어 하네스를 재사용함 |
| `docs/responses-api.md` | 백그라운드 응답의 cancel 동작 |

커밋은 `60a3f67f`(cancel 라우트와 스토어 메서드)와 `d0a76f88`(queued cancel 시 작업 중단, debug 로그, `Release` 순서)입니다.

## 3. 기술적 선택과 그 이유

### 3.1 라우팅: 백그라운드 분기를 먼저

`cancel_response`는 다른 무엇보다 먼저 `cancel_background_response`를 호출합니다. 이 함수는 모르는 id와 `background`가 `true`가 아닌 저장 응답에는 `None`을 돌려주고, 그러면 기존 포그라운드 코드가 그대로 실행됩니다. `responses_cancel_route_tests.rs`도 수정 없이 통과합니다. 활성 작업은 저장된 엔트리에도 없고, 종료 기록 직전의 짧은 구간에는 in-flight 레지스트리에도 없기 때문에 백그라운드 분기가 먼저여야 합니다. 기존의 "저장소, 그다음 in-flight" 순서가 바로 일시적 404의 원인이었습니다.

### 3.2 작업 상태에 따른 세 경우

`ResponsesStore::cancel_background`는 `jobs` 뮤텍스 아래에서 토큰을 켜고, `cancel_requested`를 설정하고, in-flight 레지스트리에서 cancel을 확인(acknowledge)합니다. 그다음 스냅샷 상태로 분기합니다.

- `queued`: `record_cancelled_locked`가 작업을 제거하고 스냅샷의 `cancelled` 사본(`output: []`, `completed_at: null`)을 상태 확인과 같은 임계 구역에서 기록합니다. 그래서 확인과 기록 사이에 작업이 생성을 시작할 수 없습니다. 응답은 `BackgroundCancel::Terminal`입니다.
- `in_progress`: 작업은 활성으로 남습니다(`BackgroundCancel::Requested`). 라우트는 `wait_background_terminal`로 드라이버 자신의 종료 객체를 `BACKGROUND_CANCEL_WAIT`까지 기다립니다. 드라이버는 토큰을 토큰마다 한 번 확인하고 부분 출력을 담은 `cancelled`를 기록하므로, 이쪽이 더 쓸모 있는 객체입니다.
- 타임아웃: `force_background_cancel`이 스냅샷으로 `cancelled`(출력 없음)를 기록하고 드라이버의 늦은 기록은 no-op이 됩니다. 그 사이 작업이 끝났다면 먼저 기록된 객체를 돌려줍니다. 이 대기는 기대 경로가 아니라 상한이며, 작업이 하나의 긴 단계에 걸려 있을 때만 소진됩니다.
- 끝난 작업 또는 모르는 id: 보존된 객체는 상태와 관계없이 `Terminal`로 돌아오고, 작업도 보존 객체도 없으면 `NotFound`입니다. 라우트는 대기 전이면 `None`(포그라운드 규칙)으로, 대기 중 객체가 밀려났다면 404로 처리합니다.

### 3.3 먼저 쓴 쪽이 이기고, 응답은 기록된 객체

드라이버, `cancel_background`, `force_background_cancel`의 모든 종료 기록은 `record_terminal_locked`를 거칩니다. 이 함수는 `jobs` 가드를 놓기 전에 저장 엔트리를 넣으므로, 활성도 저장도 아닌 작업을 보는 독자가 없습니다. 라우트는 자기 응답을 따로 만들지 않고 기록된 객체를 그대로 돌려줍니다. 따라서 `finish_registration`과 `record_terminal` 사이 구간처럼 작업이 아직 `jobs`에 있어 cancel이 404 대신 기다리는 경우를 포함해, 어떤 인터리빙에서도 cancel과 GET이 어긋날 수 없습니다. `completed`, `incomplete`, `failed`가 경주에서 이기면 기존의 400 `has already finished and cannot be cancelled`를 돌려주고 GET은 바뀌지 않습니다. 이미 `cancelled`인 응답에 다시 cancel하면 저장된 객체를 돌려줍니다.

### 3.4 토큰이 아니라 `cancel_requested` 플래그

`JobRecorder::fail`은 "첫 토큰 전에 생성이 멈추고 오류를 보고함"을 `cancelled`로 바꿔야 합니다. 가장 먼저 떠오르는 신호는 작업의 토큰이지만, `AdmissionWaiter::drop`이 수용이 거절될 때 같은 토큰을 켭니다. 첫 시도는 토큰을 읽었고 드라이버 패닉을 `cancelled`로 기록했습니다. 이제 작업에 별도의 `cancel_requested` 플래그가 있고, `request_background_cancel`과 `cancel_background`만 이를 설정하며, `fail`은 `background_cancel_requested`로 읽습니다. 플래그 검사를 끄는 되돌림 확인에서 `a_cancel_that_ends_generation_with_an_error_is_recorded_cancelled`가 `failed`로 실패합니다.

감수한 비용은 cancel 요청 이후의 모든 실패가, 무관한 드라이버 패닉까지 포함해 `cancelled`로 기록된다는 점입니다. 리뷰어는 이를 MEDIUM으로 평가했습니다. 응답과 GET은 여전히 일치하고 "cancel이 이긴다"는 이슈 요구와도 맞습니다. 진단이 가능하도록 두 번째 커밋에서 원래 코드와 메시지를 debug 레벨로 로그에 남깁니다.

### 3.5 queued cancel 시 생성 전 작업 중단 (보안 HIGH)

`queued` 작업을 cancel하면 슬롯이 바로 풀리지만, 작업 태스크는 다음에 토큰을 확인할 때까지 프렐류드(비디오 프레임 추출, 자동 compaction)와 `prepare_response_run`을 계속 실행했습니다. 클라이언트가 create와 cancel을 반복하면 ffmpeg 디코딩과 compaction 패스를 슬롯의 제한 없이 얼마든지 돌려, `--responses-background-max-jobs`를 우회할 수 있었습니다.

`unless_cancelled`는 프렐류드와 준비 단계를 `JobRecorder::terminal_recorded`와 `tokio::select!`로 경주시킵니다. cancel이 들어오면 포그라운드에서 클라이언트 연결이 끊겼을 때처럼 작업 future를 버리고 cancel 등록을 정리합니다. 이미 실행 중인 compaction 요약 패스나 클립 디코딩은 그 한 단계를 마칩니다. `terminal_recorded`는 60초 재확인 상한을 둔 `wait_background_terminal`을 반복하며, 깨우는 것은 종료 기록입니다. 테스트는 `a_queued_cancel_drops_the_pre_generation_work`입니다.

### 3.6 동시성과 락 순서

문서화한 순서는 `cancel_background`에서 `jobs`, `in_flight`(읽기), ack 뮤텍스이고, 모든 종료 기록에서는 `jobs`, `inner`입니다. `in_flight`, ack 뮤텍스, `inner`를 잡은 채 `jobs`를 잡는 곳은 없습니다. 5초 대기 중에는 락을 잡지 않습니다. 대기 신호를 먼저 `jobs`에서 복제해 냅니다. 같은 id에 대한 동시 cancel은 각각 `Notified` future 하나와 연결 하나를 쓰고, 먼저 쓴 종료 기록이 이깁니다. ack와 레지스트리 엔트리는 `JobRecorder::drop`을 포함한 모든 종료 경로에서 정리됩니다. 라우트가 종료 객체를 기록한 뒤 작업이 패닉하는 경우에 이것이 필요합니다.

### 3.7 대화 추가

라우트가 `cancelled`를 먼저 기록하면 드라이버의 이후 종료 기록은 no-op입니다. sink는 추가를 건너뛸지 판단할 때 자기 `response`가 아니라 실제로 기록된 객체(`recorded`)를 검사하므로, 출력이 빈 cancel은 대화 턴을 추가하지 않습니다.

## 4. 검증

- `cargo test --profile test-fast --features cuda -p mlxcel --lib server::routes::responses -- --test-threads=1`: 89개 통과, 수정 없는 `responses_cancel_route_tests.rs` 포함. `responses_store`는 32개 통과. 백그라운드 모듈은 반복 실행했고(경합 테스트는 매번 200라운드), 구현 리뷰에서 10회 연속 실패 없이 통과했습니다.
- clippy `-D warnings`, `cargo fmt --all -- --check`, `check_env_var_docs.py`, `insert_apache_header.py --check`가 통과합니다.
- 실제 체크포인트(`qwen3-0.6b-4bit`, GB10 CUDA, curl): `in_progress` 작업의 cancel이 부분 출력(281토큰, 리뷰 수정 후 재실행에서 316토큰)과 함께 200 `cancelled`를 돌려주었고 GET과 같았습니다(첫 실행에서 `cmp`, 3450바이트). 반복 cancel은 동일했고, create 직후 cancel은 2초 뒤 GET과 같았으며, `store: false`도 같게 동작했습니다. 완료된 작업의 cancel은 400이고 GET은 변하지 않았으며, 모르는 id는 404였고 서버 로그에 오류는 없었습니다.

## 5. 미검증 및 남은 항목

- LOW: `record_cancelled_locked`가 크기를 재려고 `jobs` 뮤텍스를 잡은 채 스냅샷을 직렬화합니다(`responses_store_cancel.rs`). 스냅샷은 `instructions`, `tools`, `metadata`를 그대로 담으므로 큰 요청에서는 수 ms의 경합이 생깁니다. 수용 시점에 스냅샷 크기를 캐시하면 없앨 수 있습니다. 코드 주석은 스냅샷에 출력이 없어 저렴하다고 하지만, 이는 출력 부분에만 해당합니다.
- LOW: 포그라운드 id에 대해 `cancel_background`가 `jobs`와 `inner` 쓰기 락 아래에서 저장 응답을 복제하고, 라우트가 `store.get`으로 다시 읽습니다. 이중 읽기이며 해롭지는 않습니다.
- 타임아웃 경로(`force_background_cancel`)는 실제 5초 상수가 아니라 주입한 50 ms 대기로 테스트했고, 토큰을 무시하는 작업을 실제 체크포인트로 확인하지는 않았습니다.
- 성능 측정은 하지 않았습니다. cancel은 호출당 뮤텍스 획득을 한 번 더합니다.
- 3.4의 MEDIUM은 제거한 것이 아니라 수용한 것입니다. cancel 이후의 무관한 실패는 debug 로그에서만 보입니다.
- 라우트가 기다리는 동안 작업이 밀려나면 404가 나옵니다. 좁은 구간이며 테스트하지 않았습니다.
- #2919에서 이어짐: 백그라운드 id는 API 키 범위로 제한되지 않습니다.

## 6. 학습 포인트

- 멱등 연산의 응답이 이후 읽기와 일치해야 한다면 저장된 레코드에서 응답을 얻고 따로 만들지 않습니다. 여기서는 라우트와 GET이 같은 객체를 돌려주므로 어긋날 수 없습니다.
- 상태 확인과 그에 의존하는 쓰기는 한 임계 구역에 둡니다. queued 작업의 `cancelled` 기록을 `cancel_background` 안에서 하므로 작업이 생성을 시작할 수 있는 구간이 없어집니다.
- "클라이언트가 요청했다"를 뜻하는 플래그는 여러 경로가 켜는 토큰과 분리해야 합니다. `AdmissionWaiter::drop`이 수용 거절 때 토큰을 켜므로 이를 cancel로 읽으면 패닉을 잘못 분류합니다.
- 제한된 자원(작업 슬롯)을 풀 때는 그 자원이 제한하던 작업도 멈춰야 합니다. `select!`로 생성 전 작업을 종료 기록과 경주시키면 포그라운드 경로의 연결 끊김과 같은 방식으로 둘이 묶입니다.
- 폴백으로 쓰는 대기는 주입 가능한 시간을 가져야 타임아웃 분기를 5초 대신 50 ms로 테스트할 수 있습니다.
