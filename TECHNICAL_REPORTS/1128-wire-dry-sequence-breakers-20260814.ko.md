# 기술 보고서: PR #1128 - fix(server): wire --dry-sequence-breaker through to the sampler

**작성일**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: Medium (무력했던 플래그가 실제로 동작하게 되고, 표현할 수 없는 값은 이제 startup을 실패시킨다)

---

## 요약

PR #1128은 이슈 #1103을 닫는다. #1102 / #1109 / #1103 체인의 세 번째이자 마지막이다. 두 서버 바이너리 모두 `--dry-sequence-breaker`를 노출하는데, 값은 CLI에서 `ServerStartupConfig`까지 흘러간 뒤 멈췄다. 토큰화 단계도, `ServerConfig` 필드도, `build_server_generate_options`의 폴백도, `/props` 항목도 없었다. 플래그는 받아들여지고 저장되고 한 번도 읽히지 않았다.

배선은 단순 복사가 아니다. 플래그는 토큰 문자열을, 샘플러는 토큰 ID를 받기 때문이다. 변환에는 tokenizer가 필요하고, 그것은 서버 config를 만드는 시점에 아직 로드돼 있지 않으므로 해결 지점을 의도적으로 골라야 했다. 토큰 하나로 표현되지 않는 breaker는 이제 조용히 버려지는 대신 startup을 실패시킨다.

---

## 1. 문제 정의

### 1.1 배경

이 체인의 세 이슈는 같은 결함을 세 위치에서 본 것이다. #1102는 요청 단위 breaker 목록이 요청 계층은 통과하고 `build_sampling_config`의 greedy 분기에서 떨어진 문제였다. #1109는 그 플래그의 이름이 두 서버 바이너리 사이에서 갈라진 문제였다. #1103은 운영자 쪽 절반이다. 플래그가 어느 방향에서도 샘플러에 도달하지 못했으므로, 요청이 물려받을 서버 전역 기본값 자체가 없었다.

### 1.2 기존 문제점

- **자취가 구조체 필드에서 끊겼다.** `grep -n dry_sequence_breakers src/server/startup.rs`는 선언과 `Default` 초기화만 돌려줬다. `ServerConfig`는 `default_dry_multiplier`, `default_dry_base`, `default_dry_allowed_length`, `default_dry_penalty_last_n`을 선언하고 거기서 멈춘다.
- **`build_server_generate_options`의 한 줄만 이웃 넷과 모양이 달랐다.** `dry_sequence_breakers: overrides.dry_sequence_breakers.unwrap_or_default()`인데, 인접한 DRY 필드는 전부 `unwrap_or(config.default_*)`다. 이 모양 차이가 결함을 눈에 보이게 만든 지점이다. 되돌아갈 서버 기본값이 없으므로, 필드를 생략한 요청은 운영자가 무엇을 넘겼든 항상 빈 벡터를 받았다.
- **실패가 양방향으로 조용했고, 두 번째 방향이 위험하다.** 플래그가 무력하다고 알려주는 것이 없었다. 그리고 breaker는 무언가를 켜는 것이 아니라 DRY 역방향 매칭을 끝내는 조건이므로, breaker 없이 DRY를 돌리면 `match_len`이 의도한 경계를 넘어 자라고 적용되는 페널티는 설정한 값 이상이 된다. `--dry-multiplier 0.8 --dry-sequence-breaker '\n'`을 준 운영자는 그 숫자가 기술하는 것보다 약한 페널티가 아니라 강한 페널티를 얻는다.
- **`/props`가 보고할 수 없었다.** `src/server/routes/props.rs`는 가진 DRY 필드 넷을 나열했다. 서버가 무엇으로 해석했는지 확인하려고 운영자가 볼 유일한 엔드포인트가 그 질문에 답할 수 없었다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| 서버에서 튜닝한 DRY가 설정과 다르게 동작하고, 그 차이가 모델 탓으로 돌려짐 | Medium | Medium |
| startup이 깨끗했고 `/props`에 다른 DRY 필드 넷이 보였으므로 운영자가 플래그가 "먹혔다"고 확인함 | Low | High |
| 나중에 죽은 표면으로 판단해 플래그를 제거하면서 이를 넘기던 스크립트가 깨짐 | Low | Low |

---

## 2. 기술적 검토 사항

### 2.1 진짜 설계 문제는 해결 지점이었다

`ServerConfig`는 `build_server_config`가 만들고, 그 함수는 모델 tokenizer가 로드되기 전에 돈다. 토큰 문자열에서 토큰 ID로의 변환에는 그 tokenizer가 필요하다. 배치 가능한 위치는 셋이었다.

1. `build_server_config`에서 토큰화한다. 기각. 자체 tokenizer를 로드해야 하므로 작업이 중복되고, 지금은 없는 파일시스템 의존성이 함수에 생긴다.
2. tokenizer 로드를 앞으로 옮긴다. 기각. `run_server`에는 현재 로드 지점보다 앞에 `serve_remote_pipeline_stage` 조기 반환이 있어서, 옮기면 원격 파이프라인 스테이지가 쓰지도 않을 tokenizer를 로드하게 된다.
3. `build_server_config`에서는 비워 두고, `run_server`가 `load_tokenizer` 직후에 채운다. 선택.

세 번째에는 실제 위험이 있다. 생성 시점과 이후 대입 사이에 필드가 유효하지 않다. 타입이 아니라 선례와 문서로 완화한다. 같은 모양이 이 함수 안 `config.pipeline_parallel_runtime`에 이미 있고, `request_options`의 `sampling.loop_detection`과 `sampling.xtc_special_token_ids`에도 있다. 셋 다 값을 결정하는 정보가 보이게 된 뒤에 해결된다. `ServerConfig`의 필드 문서 주석이 어디서 채워지는지 적고, `build_server_config` 쪽에는 왜 비어 있는지를 적은 주석이 붙는다.

### 2.2 단일 토큰 요구는 정책 선택이 아니다

샘플러의 breaker 검사는 `Vec<i32>`에 대한 `config.dry_sequence_breakers.contains(&window[p1])`이다. 다중 토큰 breaker는 그 타입에 표현 자체가 없다. llama.cpp는 토큰에서 후보 꼬리로 가는 맵을 만들어 다중 토큰을 지원하는데, 그것을 맞추는 일은 배선 변경이 아니라 샘플러 변경이고, 읽히지도 않는 플래그에 관한 이슈의 범위 밖이다. 이 데이터 모델에서 선택지는 거부하기, 조용히 버리기, 운영자가 요청하지 않은 여러 breaker로 확장하기뿐이었다. 오해를 낳지 않는 것은 거부뿐이다.

토큰 0개로 인코딩되는 문자열도 같은 부류이며 동일하게 처리된다. 에러는 문제의 문자열을 지목하고, 실제로 나온 개수를 보고하고, 플래그 이름을 적는다.

### 2.3 escape 처리는 편의가 아니라 필수다

이슈가 요구하지 않았지만 구현이 뺄 수 없었던 부분이다. `--dry-sequence-breaker '\n'`은 `\`와 `n` 두 문자로 프로세스에 도착한다. 흔한 셸 중 작은따옴표 안에서 escape를 확장하는 것은 없고, POSIX 셸에서 `"\n"`도 같은 두 문자다. 그리고 이 플래그의 헬프 텍스트는 추가된 이래 `"\n"`과 `"\t"`를 예시로 광고해 왔다.

따라서 escape 처리가 없으면 fail-closed 규칙이 이 플래그 자신의 문서화된 사용법을 startup에서 거부한다. 고치려던 버그보다 나쁜 결과다. 플래그가 조용히 무력한 상태에서, 자기 헬프의 예시에 대해 부팅을 거부하는 상태로 옮겨 가는 셈이다. 다른 쪽 실패는 더 조용하고 더 나쁘다. 리터럴 두 문자 `\n`이 어떤 어휘에서 우연히 단일 토큰이면, 서버는 개행과 아무 상관 없는 breaker를 설치한다.

규칙은 `\n`, `\t`, `\r`, `\\`을 해석하고 그 밖의 모든 백슬래시 시퀀스를 타이핑한 그대로 보존한다. 알 수 없는 시퀀스를 거부하지 않고 보존하면 실제로 백슬래시를 포함하는 breaker를 표현할 수 있고, `\\`은 그 넷 중 하나 앞에 오는 리터럴 백슬래시를 위한 탈출구다.

### 2.4 `Some(vec![])`과 `None`은 뭉개지면 안 된다

폴백은 `unwrap_or_else(|| config.default_dry_sequence_breakers.clone())`이다. 부재한 요청 필드는 서버 기본값을 물려받고, 명시적으로 빈 요청 목록은 그 요청에 한해 기본값을 끈다. 비어 있는지 검사하는 대신 `unwrap_or_else`를 쓴 이유가 이것이고, `an_explicitly_empty_request_breaker_list_disables_the_server_default`가 그것을 고정한다. 둘을 뭉개면 요청 단위로 서버 기본값을 해제할 방법이 사라진다.

### 2.5 테스트 픽스처는 빌려 오지 않고 만들어야 했다

`MlxcelTokenizer::stub_with_byte_fallback()`이 당연한 후보였고 여기서는 쓸 수 없다. BPE 모델에 merge가 없어서 `Hello`가 어휘 항목이 아니라 아무것도 아닌 것으로 토큰화되고, byte-fallback 문자 말고는 "정확히 한 토큰"을 표현하지 못한다. 테스트는 단일 문자(`a`, `b`, 개행, 탭, 공백) 어휘에 merge도 byte fallback도 없는 tokenizer를 만든다. 그러면 resolver가 구분해야 하는 세 결과가 체크포인트 없이 모두 도달 가능하다. 한 토큰, 여러 토큰, 없음. `fixture_tokenizer_behaves_as_the_tests_assume`가 그 인코딩들을 고정하므로, 픽스처가 바뀌면 실제 단언이 이상해 보이는 대신 거기서 그렇게 말하며 실패한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 제거가 아니라 배선

| 옵션 | 장점 | 단점 |
|---|---|---|
| **선택: 배선한다** | llama.cpp 호환은 명시된 목표이고 플래그는 두 바이너리의 헬프 텍스트에 있다. 조용하면서 더 강해지는 실패 모드가 사라진다 | 이미 플래그를 넘기는 배포의 생성 출력이 바뀌고, 이제 부팅을 거부할 수 있다 |
| 플래그를 제거한다 | diff가 가장 작고 startup 실패 위험이 없다 | 이를 넘기던 스크립트가 깨지고, 어차피 `CHANGELOG` 항목이 필요하며, 문서화된 llama.cpp 호환 손잡이를 버린다 |
| 배선하되 잘못된 breaker에 실패 대신 경고 | 절대 부팅을 거부하지 않는다 | 원래 결함을 축소판으로 재현한다. startup 로그의 경고야말로 아무도 읽지 않는 것이고, 결과 페널티는 다시 설정보다 강해진다 |

세 번째가 솔깃한 선택지이며, 이슈가 자세를 명시한("silently dropped or expanded 대신 명확한 메시지로 startup 실패") 이유이기도 하다. 오설정이 로그 한 줄로 끝나는 플래그는 오설정인 채로 배포되는 플래그다.

### 3.2 resolver를 자체 모듈에 둔다

`src/server/startup.rs`는 이미 크고, resolver는 tokenizer와 문자열 목록의 순수 함수이며 자체 실패 분류를 갖는다. `src/server/cors.rs`가 이 트리에서 확립된 모양이다. 운영자 입력을 검증하고, 문제의 값을 지목하고, 자체 `#[path]` 테스트 모듈을 갖는 작은 비공개 모듈. 그것을 따르면 두 검증기가 같은 종류로 알아보인다.

### 3.3 `AppState`를 세우는 대신 `default_generation_settings`를 뽑는다

`/props`에 필드가 하나 늘었고, 보고되는 필드 집합은 계약이다. 서버가 무엇으로 해석했는지 운영자가 읽는 대상이기 때문이다. axum 핸들러를 통해 단언하려면 `AppState`가 필요하고 그러려면 모델이 필요하다. 페이로드 구성을 `&ServerConfig`의 함수로 뽑으면 필드 집합을 직접 단언할 수 있고, 테스트 셋이 해석된 값, 존재하지만 빈 경우, DRY 다섯 필드 전체를 덮는다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 12 (신규 3) |
| 신규 모듈 | `src/server/dry_breakers.rs` (136줄) |
| 테스트 추가 | 17 |

### 영역별 변경

| 영역 | 파일 | 내용 |
|---|---|---|
| Resolver | `src/server/dry_breakers.rs` (신규) | `resolve_dry_sequence_breakers`와 `unescape_breaker`. 정확히 한 토큰이 아닌 breaker, 쓸 만한 것을 하나도 만들지 못한 플래그, `i32`에 들어가지 않는 토큰 id에 대해 fail-closed |
| Config | `src/server/config.rs` | `default_dry_sequence_breakers: Vec<i32>`. `build_server_config`가 아니라 `run_server`가 채운다는 문서 주석과 `Default` |
| Startup | `src/server/startup.rs` | `build_server_config`는 이유를 적고 비워 둔다. `run_server`가 `load_tokenizer` 뒤에 해결하고 해석된 ID를 로그로 남긴다 |
| 요청 경로 | `src/server/request_options.rs` | `unwrap_or_default()`가 `unwrap_or_else(|| config.default_dry_sequence_breakers.clone())`로 |
| 엔드포인트 | `src/server/routes/props.rs` | `dry_sequence_breakers` 추가. 페이로드 구성을 `default_generation_settings`로 분리 |
| 헬프 텍스트 | `src/main.rs`, `src/bin/mlx_server.rs` | 두 바이너리에서 바이트 동일(#1109 parity 단언이 요구). 서버 전역 기본값, 요청 단위 override, 단일 토큰 요구, 해석되는 escape |
| 테스트 | `dry_breakers_tests.rs`, `props_tests.rs`, `request_options_tests.rs` | resolver, `/props` 필드 집합, 기본값/override/해제 삼종에 걸쳐 17개 |
| 체인지로그 | `CHANGELOG.md` | 두 동작 변화(startup이 실패할 수 있다는 점 포함)를 기록한 `### Fixed` 항목 |

### 관련 커밋

| Hash | Type | Message |
|---|---|---|
| `470ecf5b` | fix | fix(server): wire --dry-sequence-breaker through to the sampler |

---

## 5. 검증과 후속

### 통과

- `cargo test --profile test-fast --features cuda --lib server::dry_breakers`: 11 passed.
- `--lib server::routes::props`: 3 passed.
- `--lib server::request_options`: 38 passed (35에서 증가).
- `--lib server::cli_input`: 93 passed.
- `--lib execution::sampling`: 3 passed.
- `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.

### 이번 변경에서 온 것이 아닌 기존 실패

`server::startup::muse_glimmer_startup_guard_tests::muse_glimmer_startup_allows_baseline_and_keeps_video_disabled`는 형제 테스트와 함께 돌면 실패하고 단독으로는 통과한다. `muse_glimmer_startup_rejects_xla_backend_selection`이 crate 전역 env 락 아래에서 `MLXCEL_BACKEND=xla`를 설정했다가 복원하는데, baseline 테스트는 그 락을 잡지 않으므로 그 창 안에서 변수를 관측할 수 있다. 해당 파일은 #1101이 마지막으로 손댔고 이번 diff에 없다.

### 다루지 않은 것

실제 체크포인트 검증은 없다. 주어진 breaker 문자열이 한 토큰인지는 로드된 어휘의 성질이므로, 체크포인트 테스트는 resolver가 아니라 그 체크포인트에 관한 사실을 단언하게 된다. 인수 조건은 전부 설정 경계에 있고 테스트도 거기에 있다.

페널티 값이 달라지는지 확인하는 종단 단언도 없다. 모델과 토큰 히스토리가 필요하고 시드에 의존한다. #1102가 같은 이유로 같은 판단을 기록했다.

### 후속

다중 토큰 breaker는 결정이 아니라 데이터 모델 때문에 여전히 미지원이다. 지원하려면 `SamplingConfig::dry_sequence_breakers`를 `Vec<i32>`에서 토큰-꼬리 구조로 바꿔야 하고(llama.cpp가 하는 방식), 그것은 배선이 아니라 샘플러 변경이다.

`src/commands/generate.rs`의 범위 주석은 CLI 페널티가 서버에서 같은 설정이 만드는 것보다 강하다고 적는다. 서버 기본값이 마침내 존재하므로 그 진술은 이전과 다른 방식으로 참이 됐다. 문구는 그대로도 정확해서 바꿀 필요는 없지만, 진리값이 옮겨 갔다는 점은 알아 둘 만하다.
