# 기술 보고서: PR #1736 - fix(internlm2): InternLM2 계열 rope_scaling 테스트와 NTK 재계산 로그 추가

**날짜**: 2026-09-10
**작성**: mlxcel maintainers
**검토**: 구현 리뷰 사이클
**상태**: 완료 (모듈 테스트, clippy, fmt 통과. `mlx-community/internlm2_5-7b-chat-4bit`으로 실제 체크포인트 두 번 실행. 이 계열에는 교정된 롱컨텍스트 레퍼런스가 없어 회전 자체는 검증하지 못했다)
**언어**: Rust
**위험도**: 낮음 (연산은 하나도 바뀌지 않았다. 핫 패스에 들어간 유일한 변경은 디버그 진단을 감싸는 레벨 검사 하나다)

---

## 요약

이슈 #1320은 InternLM2가 `rope_scaling`과 `rope_traditional`을 반영하게 해 달라고 요구했다. 그런데 이 브랜치를 딸 시점에 코드는 이미 그렇게 동작하고 있었다. PR #1389가 #1324를 닫으면서 `ModelArgs`와 `Attention`을 공용 `DynamicNtkRope` 스케줄에 이미 연결해 두었다. 그 범위가 #1320 구현 계획 전체와 겹친다. 남은 것은 이슈의 나머지 절반이었다. 설정 배선이 헬퍼까지 도달하는지 증명하는 계열 단위 테스트, 재계산된 base를 알려주는 디버그 로그, 실제 체크포인트 실행이다.

오래 남을 부분은 진단 로그다. `rope_scaling` 블록이 누락돼도 `max_position_embeddings` 이내 시퀀스에서는 스케줄이 정확하므로 32768토큰보다 짧은 프롬프트로는 결함을 관측할 방법이 없다. 이 PR 전까지 실행 중인 바이너리는 자기가 어떤 base로 회전하는지 아무것도 알려주지 않았다. 이제 34021토큰 프롬프트는 디버그 레벨에서 `seq_len=34021 base_eff=1077737.0` 한 줄을 남긴다. 이 값은 닫힌 형태를 그 길이에서 계산한 결과와 같다.

---

## 1. #1320이 요구한 것과 남아 있던 것

### 1.1 구현 계획은 이미 main에 있었다

이슈의 계획은 네 단계다. `ModelArgs`에 `rope_scaling`을 선언하고 `Attention`의 `rope_dims` / `rope_base` 쌍을 `DynamicNtkRope`로 바꾼다. 그다음 하드코딩된 `fast_rope` 호출 두 개를 `self.rope.apply(...)`로 교체하고 헬퍼의 `// Used by:` 주석에 계열을 추가한다. 네 단계 모두 #1389에서 이미 머지됐다. 헬퍼 하나가 두 계열을 함께 담당하고 두 계열이 mlx-lm에서 같은 잘못된 식(`rope_scale = 1 / factor if rope_type == "linear" else 2.0`을 NTK factor로 재사용)을 그대로 물려받았기 때문에 #1389는 InternLM 두 계열을 한 번에 고쳤다.

코드를 쓰기 전에 `main`을 먼저 읽은 것이 이 작업에서 가장 중요했다. 이슈 본문은 여전히 손대지 않은 상태처럼 읽힌다. 본문을 그대로 따라갔다면 아무것도 바뀌지 않는 diff가 나오거나 같은 `rope_scaling` 블록을 읽는 두 번째 파싱 경로가 생겼을 것이다.

### 1.2 남아 있던 절반

이슈의 검증 절은 유닛 테스트 다섯 개를 지정하고 `max_position_embeddings`를 처음 넘는 forward에서 `base_eff`를 디버그로 남기는 실제 체크포인트 실행을 요구한다. #1389가 실어 보낸 테스트는 세 개(체크포인트 블록 파싱, 블록 부재 시 무배율 스케줄, 미구현 스킴 로드 실패)이고 로그는 없었다. 이 PR은 `src/models/internlm2_tests.rs`에 다섯 개, `src/models/dynamic_ntk_rope_tests.rs`에 하나를 더하고 진단 로그를 넣는다.

---

## 2. 계열 자체의 설정 테스트가 필요한 이유

InternLM2의 결함은 InternLM3보다 한 단계 앞에 있었다. InternLM3은 블록을 파싱한 다음 잘못 해석했으므로 헬퍼 테스트로 틀린 답을 볼 수 있었다. InternLM2는 필드 자체를 선언하지 않았다. `{"type": "dynamic", "factor": 2.0}`은 스케줄이 만들어지기 전에 serde 단계에서 버려졌다. `DynamicNtkRope`만 따로 아무리 시험해도 이 단계는 걸리지 않는다. `config.json` 발췌에서 출발해 `ModelArgs::rope()`를 거치는 테스트라야 덮인다.

새로 넣은 다섯 개는 연산이 아니라 배선을 겨냥한다. 같은 기하의 연산 검증은 `dynamic_ntk_rope_tests.rs`가 이미 하고 있다.

| 테스트 | 고정하는 것 |
|---|---|
| `a_linear_block_reaches_the_helper_as_an_inverse_factor_scale` | `{"type": "linear", "factor": 4.0}`이 `scale() == 0.25`로 풀리고 `base_for`는 56에서도 65536에서도 움직이지 않는다 |
| `rope_traditional_reaches_the_helper` | `rope_traditional: true`가 `DynamicNtkRope::traditional()`까지 도달하고 설정이 침묵하면 serde 기본값 `false`가 유지된다. 공개 체크포인트가 실제로 쓰는 값이 이쪽이다 |
| `the_checkpoint_configs_dynamic_schedule_matches_the_issue_table` | 56, 32768, 40000, 65536 길이에서 `base_for`가 이슈 표와 상대오차 1e-3 안에서 일치한다 |
| `the_rope_type_spelling_resolves_identically_to_type_for_this_family` | 두 키 표기 모두 `Dynamic { factor: 2.0 }`을 준다. 키 이름을 바꾼 변환본이 스케줄을 조용히 바꾸지 못한다 |
| `a_dynamic_block_without_a_factor_is_a_load_error_naming_the_family` | 거부 메시지에 `internlm2`가 들어간다. 체크포인트를 여러 개 올린 운영자가 어느 쪽이 잘못됐는지 알아낸다 |

두 번째 테스트를 위해 `traditional()` 접근자를 새로 열었다. 이 PR이 추가한 유일한 공개 API다.

---

## 3. 재계산 진단 로그

`DynamicNtkRope::apply`는 이제 `base_eff`를 지역 변수로 계산해 예전과 똑같은 값을 `fast_rope`에 넘기면서 같은 값을 디버그 이벤트에도 전달한다. 이 경로의 결정 네 가지를 기록해 둔다.

### 3.1 프로세스 단위가 아니라 스케줄 단위 중복 제거

한 프로세스가 모델을 여러 개 들고 있는 상황은 흔하다. 서버의 `--models-dir` 라우팅, 파이프라인 스테이지 실행기, 텐서 병렬 랭크가 모두 바이너리 하나 뒤에 체크포인트 여럿을 놓는다. 프로세스 전역 `Once`를 쓰면 경계를 처음 넘은 체크포인트 하나 때문에 이후 어떤 체크포인트도 줄을 남기지 못한다. `rope_utils::report_unusable_rope_scaling_once`가 같은 결론에 먼저 도달해 `(model_label, rope_type)`으로 집합 키를 잡아 두었다.

`DynamicNtkRope`는 `Copy`이고 라벨을 들고 다니지 않으므로 여기서는 스케줄 자체가 키다. 키는 `(dims, base.to_bits(), max_position_embeddings, factor.to_bits())`이다. `f32` 값이 아니라 원시 비트를 쓴 이유는 전순서 래퍼 없이 튜플을 `Ord`로 만들기 위해서다. 그래서 이 키는 값이 아니라 비트 패턴의 동일성이다. `from_scaling`이 `factor`는 `is_usable_scalar`로 거르지만 `base`는 거르지 않으므로 `-0.0`이나 NaN base가 들어오면 스케줄 하나가 로그 두 줄로 갈라진다. 실제 체크포인트에서 그런 값은 나오지 않고 나오더라도 대가는 중복된 로그 한 줄이다.

`Copy`라는 사실은 레이어마다 다시 만든 값이 파라미터만 같은 별개 값이라는 뜻이기도 하다. 테스트는 그렇게 재생성한 스케줄이 다시 로그를 남기지 않는지 확인한다.

### 3.2 레벨 검사가 헬퍼가 아니라 apply에 있는 이유

경계를 넘은 뒤에는 중복 제거용 락을 호출마다 잡게 되고 `apply`는 forward 한 번에 레이어당 두 번 실행된다. 줄이 이미 나간 뒤의 정상 상태는 '이미 있음'만 답할 수 있는 조회다. 경합 없는 상태에서 40ns 정도이고 forward당으로는 수 마이크로초다. 이 컨텍스트 길이에서 디코드 한 스텝은 수십 밀리초다. 작은 값이지만 롱컨텍스트 스텝마다 모든 실행이 지불하며 구독자가 없는 실행도 예외가 아니다.

그래서 `apply`가 락을 건드리기 전에 `tracing::enabled!(tracing::Level::DEBUG)`를 먼저 묻는다. 이 검사를 `log_dynamic_rescale_once` 안으로 넣을 수는 없었다. 유닛 테스트가 구독자 없이 그 함수를 직접 호출하고 반환값을 검사하기 때문이다.

### 3.3 가드는 필드를 선언하지 않고 이벤트는 여섯 개를 선언한다

받아들이고 넘어간 빈틈이 하나 있다. 필드를 선언하지 않은 `tracing::enabled!`와 그 아래의 `tracing::debug!`는 레벨과 타깃은 같지만 필드가 다르다. 필드로 거르는 `EnvFilter` 지시자(예를 들어 `RUST_LOG=[{seq_len}]=debug`)를 주면 이벤트는 켜지고 가드는 켜지지 않아서 아무 표시 없이 줄이 사라진다. 문서화된 진입 경로는 전부 레벨 지시자다. `-v`와 `--verbosity 4`는 `server::logging::filter_directive_for_verbosity`를 거쳐 레벨 지시자로 펼쳐지고 이슈가 적어 둔 방법도 `RUST_LOG=debug`다. 빈틈은 막는 대신 소스 주석에 적어 두었다.

### 3.4 기록되는 base는 스냅샷이다

키에서 `seq_len`은 일부러 뺐다. 경계를 넘은 뒤 컨텍스트가 더 길어진 요청은 base를 다시 계산하면서도 로그를 남기지 않는다. 그 줄을 '이번 실행이 쓴 base'로 읽으면 계속 진행한 요청에서는 틀린다. 34021토큰 지점에서 base는 토큰당 62 정도씩 움직이므로 어긋남이 금세 커진다. 문서 주석이 이 점을 명시해 둔다. `seq_len`까지 키에 넣는 대안을 쓰지 않은 이유는 길이마다 한 줄씩 남겨 검증 보조 도구가 스텝별 출력으로 바뀌기 때문이다.

작은 항목이 하나 더 있다. 뮤텍스 가드는 이벤트가 나가기 전에 해제되므로 패닉을 일으키는 구독자가 락을 오염시키지 못하고 그럼에도 `unwrap_or_else(|err| err.into_inner())`로 복구한다. 어느 쪽이든 집합은 유효한 집합으로 남고 이후 모든 줄을 잃는 쪽이 더 나쁘다.

---

## 4. 문서 수정 세 가지

**존재하지 않는 체크포인트 디렉터리.** `models/internlm2-7b-4bit`는 모듈 헤더, 테스트 파일 두 개, `docs/supported-models.md`에 모두 적혀 있었다. 출처는 같은 이름을 쓴 이슈 본문이다. 실제 디렉터리는 `models/internlm2_5-7b-chat-4bit`다. 이슈의 검증 명령을 그대로 따라간 사람은 이유를 알려 주는 것 없이 경로 없음만 만났을 것이다.

**증거 수준을 잘못 주장하던 항목.** `docs/supported-models.md`의 항목은 InternLM 2와 3을 함께 다루면서 둘 다에 InternLM3의 증거를 붙여 놓았다. 교정된 mlx-lm 0.31.3 그리디 레퍼런스와 토큰 단위로 일치하고 `tests/causal_prefill_greedy_parity.rs`가 고정한다는 내용이다. InternLM2에는 그런 레퍼런스가 없다. 항목은 이제 자기 체크포인트가 실제로 보인 것만 적는다. 34021토큰 프롬프트가 dynamic 분기에 들어가 문서화된 base로 재계산한다는 사실, 그리고 함께 실행한 바이트 동일성 쌍이 패리티 결과가 아니라 대조군이라는 사실이다.

**길이를 잘못 잡고 비교한 f32 / f64 수치.** 이전 괄호 설명은 34020에서 계산한 닫힌 형태를 34021에서 출력된 f32 값과 비교하고 그 차이를 정밀도 차이로 적었다. 실제로는 길이 차이다. f64로 34021에서 계산하면 1077736.99가 나오므로 코드가 계산하는 f32는 로그가 출력하는 모든 자리에서 일치하고 34022는 1077799.125를 주는데 같은 f32 계산이 비트 단위로 재현한다. 이제 항목은 값과 길이를 함께 적는다.

---

## 5. 검증과 검증하지 못한 것

### 5.1 게이트

| 게이트 | 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib models::internlm2` | 8 passed |
| `cargo test --profile test-fast --features metal,accelerate --lib models::dynamic_ntk_rope` | 17 passed |
| `cargo clippy --lib --tests`, `cargo fmt --all --check` | clean |
| PR의 CI (검사 13개) | 통과 |
| `cargo test --workspace --profile test-fast --features metal,accelerate` | 이 PR이 아니라 `nightly-verify.yml`에서 실행 |

마지막 행은 #1320의 인수 조건 중 하나다. `ci.yml`은 범위를 좁힌 clippy와 fmt 잡을 돌리므로 워크스페이스 게이트는 이 PR의 검사가 아니라 나이틀리 잡이 만족시킨다.

### 5.2 실제 체크포인트

`models/internlm2_5-7b-chat-4bit`, 그리디, 이 브랜치의 부모 커밋으로 빌드한 바이너리와 대조했다.

| 프롬프트 | 생성 토큰 | 결과 |
|---|---|---|
| 89토큰 | `-n 64` | 바이트 동일, MD5 `5a6d6133` |
| 34021토큰 | `-n 32` | 바이트 동일, MD5 `69a12bff` |

이 쌍은 패리티 결과가 아니라 대조군이다. 부모 커밋에 이미 #1389의 배선이 들어 있으므로 두 갈래의 차이는 이 PR이 더한 디버그 진단뿐이고 바이트 동일성은 그 진단이 출력에 영향을 주지 않는다는 사실만 보인다. 회전에 관한 증거로 쓰지 않고 그대로 적어 둔다.

롱컨텍스트 실행은 `tracing::enabled!` 게이트를 넣은 뒤 최종 바이너리로 다시 돌렸다. 그 커밋 이전에만 동작했던 것이 아니라 지금도 `RUST_LOG=debug` 실행에 진단이 도달한다는 뜻이다. 34021토큰 프롬프트는 `seq_len=34021 base_eff=1077737.0` 한 줄만 남긴다. 생성 결과는 프롬프트 끝의 질문에 레일리 산란을 주제로 유창하게 답한다.

### 5.3 입증되지 않은 것

회전 자체다. InternLM3을 교정된 mlx-lm 레퍼런스에 붙일 수 있었던 이유는 그런 레퍼런스가 존재하기 때문이다. 원본 mlx-lm 0.31.3은 `dynamic` 체크포인트에서 모든 토큰을 실제 위치의 두 배로 회전시키므로 레퍼런스는 레이어별 rope 모듈만 교체하고 나머지는 그대로 두어 만들어야 했다. 이번에 InternLM2용으로 같은 레퍼런스를 만들지는 않았다. 이 계열이 현재 주장하는 것은 문서화된 길이에서 dynamic 분기에 들어가 문서화된 base를 계산한다는 것까지다. 그 이상은 아니다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 | 5 |
| 추가된 라인 | 293 |
| 삭제된 라인 | 10 |
| 추가된 테스트 | 6개 (계열 단위 5, 중복 제거 1) |

### 파일별 변경

- `src/models/dynamic_ntk_rope.rs`: `traditional()` 접근자, 스케줄 단위 중복 제거 집합을 쓰는 `log_dynamic_rescale_once`, `apply`의 `tracing::enabled!` 게이트. 회전 연산은 그대로다. `base_for(seq_len)`을 지역 변수로 끌어올려 같은 `fast_rope` 호출에 넘긴다.
- `src/models/internlm2_tests.rs`: `ModelArgs::rope()`를 덮는 테스트 다섯 개, 그리고 어느 테스트가 #1324에서 왔고 어느 테스트가 #1320에서 왔는지 적은 헤더 주석.
- `src/models/dynamic_ntk_rope_tests.rs`: `the_rescale_log_fires_once_per_schedule_not_once_per_process` 테스트, 그리고 그 테스트가 의존하는 기하를 예약해 두는 파일 단위 주석.
- `src/models/internlm2.rs`: #1389의 배선과 #1320의 테스트·로그를 구분하는 모듈 헤더 문단, 그리고 교정된 체크포인트 디렉터리.
- `docs/supported-models.md`: InternLM 항목의 증거 주장과 `base_eff` 수치.

### 관련 커밋

| Hash | Type | Subject |
|---|---|---|
| `c26adb2` | fix | add family rope_scaling tests and NTK rescale log |
| `9d92894` | fix | key the NTK rescale log per schedule |
| `a2317d0` | perf | skip the dedup lock when debug logging is off |
| `a1accf3` | docs | state what the long-context run does and does not show |
| `1aba962` | docs | correct four claims in the rescale log's comments |
| `4d2911a` | docs | correct the f64 base_eff figure for InternLM2 |

### 관련 이슈와 PR

Closes #1320. #1324와 그 구현인 #1389에 의존한다. `src/models/dynamic_ntk_rope.rs`와 이 PR이 시험하는 배선이 거기서 왔다.

---

## 7. 후속 조치

**InternLM2용 롱컨텍스트 패리티 레퍼런스가 없다.** 만드는 방법은 InternLM3과 같다. mlx-lm 0.31.3에서 레이어별 rope 모듈만 교정된 스케줄로 바꾸고 동일한 프롬프트 id로 비교하면 된다. 그때까지는 `docs/supported-models.md` 항목이 그 사실을 적어 둔다. 완화책이지 해결은 아니다.

**3.3절의 필드 필터 사각지대.** 필드로 거르는 사용자는 줄을 조용히 잃는다. 막으려면 `enabled!` 가드에 이벤트의 필드를 선언하거나 가드를 빼고 락 비용을 받아들여야 한다.

**중복 제거 집합이 테스트 순서와 얽힌다.** `log_dynamic_rescale_once`는 프로세스 전역 집합에 쓰고 유닛 테스트는 첫 교차 동작을 검사하므로 예약된 기하로 `max_position_embeddings`를 넘는 테스트가 나중에 추가되면 첫 단언이 실행 순서에 좌우된다. 지금의 완화책은 예약 기하를 적어 둔 주석이다. 이 파일에 로그를 남기는 테스트가 더 늘어난다면 `#[cfg(test)]` 뒤의 리셋 훅이 더 튼튼하다.

### 옮겨갈 교훈

이슈는 다른 번호로 올라온 PR이 거의 다 닫아 놓기도 하고 이슈 본문에는 그런 표시가 남지 않는다. #1320은 구현 계획 전체가 `main`에 올라가 있는 동안에도 손대지 않은 상태처럼 읽혔다. #1389가 #1324를 위해 공용 헬퍼를 넣으면서 두 계열을 함께 배선했기 때문이다. 이를 잡아낸 확인 자체는 간단했다. 계획을 읽기 전에 현재 코드를 읽으면 된다. 그렇게 피한 손해는 반나절이 아니라 `rope_scaling` 블록 하나를 읽는 파싱 경로가 둘로 갈라지는 것이다. 공용 헬퍼를 도입한 이유가 정확히 그 결함을 막는 데 있었다.
