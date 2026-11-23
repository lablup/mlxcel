# PR #3301: MCP, decision, rerank 서버 요청 작업량 제한

**날짜**: 2026-11-23
**상태**: PR 열림 (#3296 해결). 변경한 모듈의 단위 테스트 통과. 실제 체크포인트는 필요하지 않았음

## 요약

서버의 네 경로가 신뢰할 수 없는 요청 작업을 제한하거나 중단할 방법 없이 받아들이고 있었습니다. `npx`나 `uvx`로 띄운 stdio MCP 서버는 런처만 종료되어 클라이언트가 사라져도 살아남았습니다. `/v1/decisions`는 비디오 개수를 제한하지 않았고, 300초 상한을 오디오 전체를 디코딩한 뒤에야 적용했으며, 타임아웃 시 504를 반환한 뒤에도 워커가 계속 실행되고 핸들러가 temp 파일 가드를 먼저 해제했습니다. `/v1/rerank`는 문서 개수를 제한하지 않았습니다.

이 PR은 다섯 항목을 한 번에 고칩니다. MCP stdio 자식 프로세스는 자체 프로세스 그룹을 갖고 그룹 전체가 종료됩니다. decision은 비디오를 8개로 제한하고, 긴 오디오를 디코딩 전에 컨테이너 헤더로 거부하며, 타임아웃 시 대기 중인 작업을 취소하고 워커가 끝날 때까지 temp 파일을 유지합니다. rerank는 `MLXCEL_RERANK_MAX_DOCUMENTS`로 문서를 1024개로 제한합니다. 보안 리뷰에서 헤더에 매우 큰 소스 샘플레이트를 선언하는 decision 오디오 구멍이 추가로 발견되어, 그 레이트를 192 kHz로 제한했습니다. 한 가지 트레이드오프는 감수합니다. MCP 자식 프로세스는 더 이상 터미널 SIGINT를 직접 받지 않습니다.

## 1. 문제 정의

### 1.1 제한이 없거나 멈출 수 없는 다섯 경로

| 항목 | 경로 또는 모듈 | 실패 양상 |
|---|---|---|
| 1 | `server/mcp/client.rs` | `kill_on_drop`과 `shutdown`은 자식에게만 시그널을 보냅니다. 런처의 손자 프로세스(실제 MCP 서버)는 요청이나 서버가 끝난 뒤에도 계속 실행됩니다. |
| 2 | `/v1/decisions` 비디오 | 개수 상한이 없습니다. 모델의 토큰 게이트가 초과를 거부하기 전에 각 비디오를 resolve(다운로드 포함)합니다. |
| 3 | `/v1/decisions` 타임아웃 | `predict`가 `Timeout`을 반환해도 큐에 남은 명령이 나중에 실행되어, 이미 504를 받은 호출자를 위해 모델이 돕니다. 핸들러가 `spawn_blocking` 직후 `PrivateTempDir` 가드를 해제하므로, 실행 중인 예측의 파일이 사라질 수 있습니다. |
| 4 | `/v1/decisions` 오디오 | 300초 상한이 24시간 정책의 `decode_wav_native_compat`로 클립 전체를 디코딩한 뒤에 적용되었습니다. |
| 5 | `/v1/rerank` | `validate_items`는 빈 목록만 거부했으므로, 요청 하나가 임의의 개수의 문서와 이미지 fetch를 담을 수 있었습니다. |

### 1.2 방치 시 위험

| 위험 | 영향 | 가능성 |
|---|---|---|
| 요청마다 MCP 손자 프로세스가 누적 | 중간: 장기 실행 서버의 프로세스와 메모리 고갈 | 런처형 서버에서 높음 |
| 요청 하나가 무제한 fetch, 디코딩, 큐 작업을 유발 | 중간: 인증된 또는 로컬 클라이언트에 의한 서비스 거부 | 중간 |
| 실행 중인 디코딩 아래에서 temp 파일 삭제 | 낮음에서 중간: 504 이후 잘못되거나 실패한 예측 | 낮음 |

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/server/mcp/client.rs` | Unix에서 spawn 시 `command.process_group(0)`을 적용하고 pgid를 `process_group: Option<i32>`에 저장합니다. `kill_process_group`은 `killpg(SIGKILL)`을 한 번 보내고 id를 `take()`합니다. `shutdown`, `has_exited`, 새 `Drop`이 이를 호출합니다. Windows는 그대로입니다. |
| `src/server/routes/decisions.rs` | `MAX_DECISION_VIDEOS = 8`을 `check_video_support`에서 resolve 전에 검사하고, 상한을 명시한 400을 반환합니다. 이 검사는 ffmpeg 확인보다 먼저 실행됩니다. 가드는 워커 호출로 이동합니다. |
| `src/server/decision_worker.rs`, `decision_model.rs` | `DecisionCommand::Predict`가 `cancelled: Arc<AtomicBool>`과 `temp_files: Vec<PrivateTempDir>`를 가집니다. `predict`는 타임아웃 시 플래그를 설정합니다. `worker_loop`는 플래그가 설정된 명령을 건너뛰고, 예측 또는 건너뛴 뒤에만 temp 파일을 해제합니다. `DecisionModelProvider::predict` 트레이트가 `temp_files`를 받습니다. |
| `src/audio/preprocessing.rs`, `preprocessing_policy.rs` | 새 `decode_native_within(bytes, max_duration_seconds)`. 먼저 `container::inspect`를 실행하여 헤더의 프레임 수가 샘플 디코딩 전에 `source duration samples` 제한을 발동시키고, 소스 레이트를 192 kHz로 제한합니다. 다른 호출자를 위한 호환 디코더는 그대로입니다. |
| `src/decision/media.rs` | `decode_audio_16k`가 `decode_native_within(MAX_DECISION_AUDIO_SECONDS)`를 호출합니다. `duration_message`가 컨테이너 제한을 기존과 같은 300초 사용자 메시지로 변환합니다. 디코딩 후 검사는 보조 장치로 유지합니다. |
| `src/server/routes/rerank.rs` | `MLXCEL_RERANK_MAX_DOCUMENTS`(기본 1024, 한 번만 읽음). 미설정, 0, 숫자가 아닌 값은 기본값을 유지하며, 뒤의 둘은 경고를 남깁니다. `validate_items`가 이미지 fetch 전에 긴 목록을 거부합니다. |
| 테스트 | `client_tests.rs`와 `tests/fixtures/mcp/fake_mcp_server.py`(shutdown 후와 drop 후 손자 프로세스 종료), `decisions_tests.rs`(비디오 9개), `decision_worker_tests.rs`(건너뛰기와 temp 디렉터리 수명), `media_tests.rs`(301초 WAV와 레이트 제한), `rerank_limit_tests.rs`(문서 1025개와 환경 변수 파싱). |
| 문서 | `docs/decisions.md`, `docs/embeddings.md`, `docs/environment-variables.md`. |

17개 파일, +763/-37. 이 저장소에는 해당 문서의 한국어 대응본이 없어 추가하지 않았습니다.

## 3. 기술적 결정

### 3.1 그룹을 한 번만 종료

`killpg`는 `shutdown`, `has_exited`(리더를 reap한 뒤), `Drop`에서 보냅니다. id는 첫 전송 때 `take()`되므로 `shutdown` 뒤의 `Drop`이 커널이 재사용했을 수 있는 그룹 id에 시그널을 보내지 않습니다. `Drop`에서만 그룹에 시그널을 보내는 대안은, `shutdown`이 먼저 자식을 reap하여 pgid가 낡아지므로 기각했습니다.

### 3.2 가드는 명령과 함께 이동

타임아웃 시 핸들러는 워커가 아직 파일을 디코딩 중인지 알 수 없습니다. `PrivateTempDir` 가드를 명령에 넣으면 워커 스레드만이 예측 후 또는 취소된 명령을 건너뛴 직후에 가드를 해제합니다. `DecisionMedia`에 별도의 `Arc` 보관자를 두는 대안은 파일 소유권을 미디어 타입과 라우트에 분산시킵니다. 프로바이더 트레이트에 `temp_files`를 전달하면 모든 테스트 페이크를 수정해야 했지만 소유권은 한 곳에 남습니다.

취소는 협력적입니다. 이미 실행 중인 예측은 중단되지 않고, 큐에 남은 명령만 건너뜁니다.

### 3.3 헤더로 긴 오디오 거부

컨테이너 프로브는 이미 선언된 프레임 수로 길이 제한을 계산하므로, `MAX_DECISION_AUDIO_SECONDS`를 정책에 넘기면 새 파싱 없이 301초 WAV를 샘플 디코딩 전에 거부합니다. `duration_message`는 `actual * cap / maximum`으로 클립 길이를 복원하여, 사용자가 보던 기존 400 문구를 그대로 냅니다.

### 3.4 소스 레이트 제한 (보안 리뷰 수정)

길이 상한만으로는 작업량이 제한되지 않습니다. 수 MHz의 샘플레이트를 선언한 헤더는 짧은 "길이"를 수억 개의 샘플로 바꿉니다. `decode_native_within`은 이제 채팅 정책의 192 kHz 상한(`MAX_SOURCE_SAMPLE_RATE`, `pub(super)`로 변경)을 소스 레이트 제한으로 전달합니다. 레거시 디코더는 `u32::MAX`를 유지하므로 decision이 아닌 호출자는 영향이 없습니다.

### 3.5 잘못된 rerank 상한은 제한 유지

0이거나 파싱할 수 없는 `MLXCEL_RERANK_MAX_DOCUMENTS`는 검사를 끄지 않고 경고와 함께 1024로 되돌아갑니다. 오타 하나로 제한이 사라지지 않습니다.

### 3.6 트레이드오프: MCP 자식은 터미널 SIGINT를 받지 않음

자체 그룹의 프로세스는 터미널의 포그라운드 그룹 밖에 있으므로, 서버 터미널의 Ctrl-C가 MCP 자식에게 직접 도달하지 않습니다. 자식은 서버 자체의 종료 경로(`shutdown` 또는 `Drop`) 또는 stdin 종료로 멈춥니다. `Drop`을 건너뛰는 서버 강제 종료는 이전처럼 자식을 남깁니다. Windows 프로세스 트리 처리는 범위 밖입니다.

## 4. 검증

좁은 셀렉터만 사용했고 `--profile test-fast --features metal,accelerate`, `RUST_MIN_STACK=16777216`, `CARGO_BUILD_JOBS=6`을 적용했습니다. 정확한 결과는 PR 인계 내용과 PR 코멘트에 있습니다. 실행한 셀렉터: `server::mcp`, `server::routes::responses_mcp`, `server::routes::decisions`, `server::decision_worker`, `decision::media`, `server::routes::rerank`, `audio::preprocessing`, `coverage_tests`. 추가로 `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --all -- --check`, CI 계약 스크립트(`check_used_by`, `check_kernel_dtype_keys`, `check_env_var_docs`, `check_crate_versions`, `check_kernel_port_dispatch`, `check_binary_assets`, `insert_apache_header --check`)를 실행했습니다.

새 테스트 다섯 개는 각각 main에 없는 동작을 검증합니다. MCP 테스트는 제한된 폴링 안에 손자 pid가 ESRCH를 반환하는지, 라우트 테스트는 400과 resolve 또는 fetch가 없음을, 워커 테스트는 타임아웃된 큐 요청이 예측되지 않고 첫 요청의 temp 디렉터리가 예측이 끝날 때까지 존재함을 확인합니다.

미실행: 워크스페이스 전체 게이트와 실제 체크포인트 스모크 실행은 오케스트레이터에게 남겼습니다.

## 5. 미검증 및 남은 항목

- **LOW (a): `has_exited`의 pid 재사용 구간.** `try_wait`가 리더를 reap한 몇 마이크로초 뒤에 `killpg`가 실행됩니다. 그 사이 커널이 pid를 새 그룹으로 재사용했다면 시그널이 그쪽에 갈 수 있습니다. 구간이 마이크로초이고 pid 카운터가 한 바퀴 돌아야 하므로 그대로 두었습니다.
- **LOW (b): 오디오 길이 메시지 문구.** 프레임 수를 선언하지 않는 압축 오디오(mp3, flac)는 컨테이너가 헤더로 거부할 수 없습니다. 디코딩 루프가 같은 상한을 적용하고, 이때 메시지는 상한에 도달한 시점의 길이를 알려 근사값처럼 읽힐 수 있습니다. 문구는 바꾸지 않았습니다.
- **LOW (c): rerank 텍스트 크기.** 쿼리와 문서 텍스트는 JSON 본문 크기로만 제한됩니다. 이 PR 이전부터 있던 문제이며 문서 개수 상한으로는 해결되지 않습니다.
- **실행 중인 예측은 중단되지 않음.** 타임아웃 시 대기 중인 작업만 건너뜁니다.
- **Windows.** Windows에서는 MCP 손자 프로세스가 여전히 종료되지 않습니다.

## 6. 학습 포인트

- 런처를 죽이는 것은 런처가 띄운 서버를 죽이는 것이 아닙니다. spawn 시 프로세스 그룹을 소유하고, 한 번 시그널을 보낸 뒤 id를 버리십시오.
- 기다리기를 멈춘 호출자는 자원 소유권을 해제하지 말고, 아직 그 자원을 쓰는 코드에 넘겨야 합니다.
- 비싼 단계 뒤에 실행되는 상한은 제한이 아닙니다. 입력이 자신의 크기를 선언하는 가장 이른 지점으로 옮기고, 선언된 모든 양(길이뿐 아니라 레이트)을 제한하십시오.

## 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `0e5ebd3b` | fix(security) | bound MCP, decision and rerank request work |
| `16310611` | fix(security) | bound the decision audio source sample rate |
