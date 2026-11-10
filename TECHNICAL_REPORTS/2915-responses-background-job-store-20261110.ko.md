# PR #2915: ResponsesStore의 백그라운드 응답 작업 추적

**날짜**: 2026-11-10
**상태**: 완료 (draft PR로 열림, #2897을 닫음, 에픽 #2903의 일부)

## 요약

PR #2915는 OpenAI Responses API 백그라운드 모드(`background: true`)의 저장소 기반입니다. 이전에는 `ResponsesStore`가 생성이 끝난 뒤의 응답만 보관했고, 진행 중인 상태는 취소 토큰뿐이었습니다. 백그라운드 요청은 생성 시점부터 폴링할 수 있어야 하므로, 이제 저장소는 진행 중인 작업을 TTL과 LRU 대상 엔트리와 분리된 두 번째 맵 `jobs`에 별도 락으로 보관합니다. 작업이 끝나면 `finish_background`가 한 번의 임계 구역 안에서 작업을 일반 엔트리 맵으로 옮깁니다. `store: false` 작업은 폴링으로는 일정 시간 읽을 수 있지만 `previous_response_id`로는 읽을 수 없는 *ephemeral* 종료 엔트리를 남깁니다. 두 플래그가 기능의 한도를 정합니다: `--responses-background-max-jobs`(기본 64)와 `--responses-background-ephemeral-ttl-secs`(기본 600, 최소 1). 이 PR은 HTTP 동작을 추가하지 않으며 라우트 연결은 #2898에서 합니다. 리뷰와 보안 점검에서 CRITICAL이나 HIGH는 없었고, 두 리뷰 모두 지적한 MEDIUM 하나(`jobs` 락을 잡은 채 두 번 직렬화)는 수정했습니다. 12개 파일, +1025/-17이며 그중 약 450줄이 테스트입니다.

## 1. 문제 정의

OpenAI 백그라운드 모드는 `queued` 응답을 즉시 돌려주고, 클라이언트가 `GET /v1/responses/{id}`로 폴링하거나 취소하거나 완료를 기다리게 합니다. 기존 저장소로는 이를 지원할 수 없었습니다.

- 생성과 완료 사이에 응답 기록이 없어서 생성 중 폴링은 404가 됩니다.
- `ResponseStatus::Queued`는 타입에만 있고 쓰이지 않았습니다.
- 엔트리는 TTL과 LRU로 축출됩니다. 오래 걸리는 작업을 일반 엔트리로 두면 실행 중에 축출될 수 있습니다.
- `store: false` 응답은 저장되지 않지만, 백그라운드 클라이언트는 결과를 폴링해야 합니다.

형제 하위 이슈(#2898 생성과 폴링, #2899, #2900, #2901)가 고정된 저장소 API를 전제로 작성되어 있어서, 이 PR은 이슈의 시그니처를 그대로 구현해야 했습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/server/responses_store.rs` | `jobs: Mutex<BackgroundState>` 맵, `BackgroundAdmitError`, `BackgroundCancel`, 백그라운드 메서드 8개, `Entry.ephemeral_ttl`, `ResponsesStoreConfig.background_max_jobs`와 `background_ephemeral_ttl`, 포그라운드와 백그라운드 삽입이 공유하는 `insert_locked` |
| `src/server/responses_store_background_tests.rs` (신규) | 테스트 13개: 한도와 슬롯 반환, 비활성 및 예산 초과 거부, 상태 전이, TTL/LRU 생존, 8스레드 종료 경합, ephemeral 노출과 만료, 과대 종료 객체, 바이트 회계, 취소, 대기 |
| `src/main.rs`, `src/bin/mlx_server.rs`, `src/commands/serve.rs`, `src/server/cli_input.rs`, `src/server/startup.rs` | 환경 변수를 갖는 플래그 2개를 `ResponsesStoreConfig`까지 전달 |
| `src/main_tests.rs`, `src/bin/mlx_server.rs` 테스트 | 두 바이너리에서 기본값, 플래그 값, 환경 변수 값, TTL 0 거부 |
| `docs/responses-api.md`, `docs/environment-variables.md` | 플래그 행, 환경 변수 행, 작업 보존과 예산 설명 |

## 3. 기술적 선택과 그 이유

### 3.1 엔트리 플래그 대신 별도 맵

진행 중인 작업을 "고정" 표시가 붙은 일반 엔트리로 둘 수도 있었습니다. 그러면 모든 축출 경로(`sweep_expired`, `evict_to_limits`)가 이를 건너뛰어야 하고, 모든 바이트 계산이 포함 여부를 판단해야 합니다. 작업을 다른 락 아래의 다른 자료구조 `jobs`에 두면 어떤 축출 코드도 이를 볼 수 없으므로 구조적으로 TTL과 LRU에서 제외됩니다. 대가는 두 번째 락이며, 고정된 순서로 처리합니다(3.2).

### 3.2 락 순서와 원자적 인계

두 락을 모두 잡는 경로는 항상 `jobs`를 먼저, 그다음 `inner`를 잡습니다. `finish_background`는 `jobs`를 잡은 상태로 작업 제거와 종료 엔트리 삽입을 모두 수행하므로, 동시에 실행되는 `background_snapshot`(역시 `jobs`를 먼저 잡음)은 진행 중 작업이나 종료 엔트리 중 하나를 보게 되고 둘 다 없는 상태는 보지 않습니다. 작업을 먼저 제거한 호출이 이기고, 이후 호출은 기록된 엔트리를 돌려받습니다. 8스레드 테스트가 모든 호출자가 같은 객체를 받는지 확인합니다.

### 3.3 깨움을 놓치지 않는 대기

`wait_background_terminal`은 작업별 `BackgroundSignal { Notify, OnceLock<ResponseObject> }`를 씁니다. `finish_background`는 `notify_waiters` 전에 `OnceLock`을 설정합니다. 대기자는 `Notified` future를 먼저 만들고 그다음 `OnceLock`을 확인합니다. Tokio는 생성된 `Notified` future가 이후의 모든 `notify_waiters` 호출을 관찰한다고 보장하므로, 확인에서 값을 보거나 future가 깨어나거나 둘 중 하나입니다. 결과를 저장소가 아닌 `OnceLock`에서 읽으므로, LRU 압력으로 엔트리가 즉시 축출되어도 대기자는 종료 객체를 받습니다.

### 3.4 축출 압력 없는 바이트 예산

진행 중 작업의 바이트는 `active_bytes`로 따로 추적하고, `approximate_total_bytes`에 더하며, 수락 시점에 검사합니다(진행 중 바이트와 새 작업의 합이 `max_bytes`를 넘으면 `OverBudget`). 이 값은 `evict_to_limits`에 쓰이지 않습니다. 실행 중 작업이 몰렸다고 완료된 기록을 밀어내면 안 되기 때문입니다. 최악의 경우 저장소는 `max_bytes`의 약 두 배(진행 중 작업과 저장 엔트리)를 보유하며, 이는 이슈가 정한 한도입니다.

### 3.5 과대 종료 객체

`max_bytes`보다 큰 종료 객체는 삽입 직후 축출되어, 작업 종료 후의 폴링이 404가 됩니다. 대신 `error.code: "server_error"`, 빈 출력, 예산을 명시한 메시지를 가진 `failed` 객체로 기록합니다. 이 기록에서는 입력 항목을 버리고, 에코된 필드(`instructions`, `tools`, `metadata`)가 여전히 넘치면 그것도 버립니다.

### 3.6 `store: false`의 ephemeral 엔트리

`store: false` 작업의 종료 객체는 `ephemeral_ttl`이 설정된 일반 엔트리입니다. `get()`은 이를 `None`으로 돌려주므로 `previous_response_id`와 기존 GET 경로는 읽을 수 없고, `background_snapshot`, `request_background_cancel`, `wait_background_terminal`은 읽을 수 있습니다. 만료는 기존 지연 sweep을 쓰며, 일반 TTL과 ephemeral TTL 중 짧은 쪽에서 만료됩니다. 이슈는 `ephemeral_until: Instant`를 제안했지만, 구현은 `Duration`을 저장하고 경과 시간을 비교해서 매우 큰 TTL에서의 `Instant` 오버플로를 없앴습니다. 비공개 필드라 형제 이슈의 API에는 영향이 없습니다.

### 3.7 기본값

제안된 기본값을 유지했습니다. 진행 중 작업마다 생성 스레드 하나를 쓰므로 64개는 Tokio의 기본 blocking 스레드 512개보다 충분히 낮습니다. ephemeral TTL 600초는 저장되지 않는 출력을 한 시간 내내 들고 있지 않으면서 클라이언트가 폴링할 시간을 줍니다. TTL 0은 플래그와 환경 변수 모두 clap(`range(1..)`)이 거부하고, `startup.rs`가 프로그램으로 들어온 0을 1로 보정합니다.

## 4. 리뷰 중 변경

두 리뷰 모두 `finish_background`가 `jobs` 뮤텍스를 잡은 채 종료 객체를 두 번(예산 확인과 삽입) 직렬화한다고 지적했습니다. 수 MB 결과라면 그 시간 동안 모든 수락, 폴링, 취소가 막힙니다. 수정은 락을 잡기 전에 종료 객체를 한 번 측정하고, 수락 시 계산한 입력 항목 크기(`BackgroundJob.input_bytes`)를 재사용하며, 미리 계산한 엔트리 크기를 `insert_locked`에 넘깁니다. 포그라운드 `insert()`도 이제 저장소 락을 잡기 전에 크기를 계산합니다.

## 5. 형제 하위 이슈를 위한 미결 사항

- `remove()`(DELETE)는 `store: false` 종료 엔트리도 삭제합니다. 이런 id에 DELETE가 무엇을 돌려줄지는 #2898이 정합니다.
- 이미 진행 중인 id를 다시 수락하면 작업이 교체되고 기존 대기자는 타임아웃됩니다. id는 서버가 만든 UUID라 실제로는 도달할 수 없습니다.
- `max_bytes`가 아주 작으면 축소한 `failed` 객체도 예산을 넘어 즉시 축출됩니다. 대기자는 받지만 이후 폴링은 404입니다.
- `background_snapshot`, `request_background_cancel`, `wait_background_terminal`은 미스일 때 `inner`를 sweep하는 동안 `jobs`를 계속 잡습니다. 먼저 놓아도 안전하지만 단순성을 위해 그대로 두었습니다.
- 작업은 메모리에만 있고 재시작하면 사라집니다.

## 6. 검증

- GB10에서 `gpu-lock` 아래 `responses_store` lib 테스트: 29개 통과(기존 16개는 단언 변경 없음, 신규 13개).
- 두 바이너리의 플래그 테스트: 3 + 3 통과. `cli_input`, `server::routes::responses`, `serve` 선택자도 통과.
- lib, 테스트, 두 바이너리에 대한 `-D warnings` clippy, `cargo fmt --check`, 환경 변수 문서와 라이선스 헤더 검사.
- 실제 바이너리(`mlxcel-server`, test-fast 프로필): `--help`에 두 플래그가 나옵니다. 플래그나 환경 변수로 TTL 0을 주면 종료 코드 2로 끝납니다. 잘못된 `MLXCEL_RESPONSES_BACKGROUND_MAX_JOBS`가 거부되므로 환경 변수를 읽는 것이 확인됩니다. 두 환경 변수를 설정하고 `models/qwen3-0.6b-4bit`로 띄운 서버가 `store: true` `/v1/responses` 요청을 처리했고(`completed`, 출력 `2 + 3 = 5.`), `GET /v1/responses/{id}`로 돌려주었습니다.

여기서 검증하지 않은 것: HTTP 백그라운드 동작(#2898 전에는 존재하지 않음)과 전체 lib 스위트(오케스트레이터가 실행).

## 7. 학습 포인트

- 두 락이 하나의 논리적 전이를 보호할 때, 고정된 획득 순서와 첫 번째 락 아래에서 전이 전체를 수행하는 방식으로 락을 합치지 않고도 읽는 쪽에 전부 아니면 전무의 관점을 줄 수 있습니다.
- `tokio::sync::Notify`의 안전한 패턴은 `notified()` 생성, 조건 확인, 대기 순서입니다. 결과를 담은 공유 `OnceLock`은 대기자를 이후의 축출과 무관하게 만듭니다.
- 호출자의 데이터에만 의존하는 비싼 작업(직렬화)은 락 밖에서 하고, 계산된 크기를 넘기는 편이 좋습니다.
