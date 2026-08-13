# 기술 보고서: PR #1124 - fix(server): thread dry_sequence_breakers through the greedy branch

**작성일**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

PR #1124는 이슈 #1102를 닫는다. `build_sampling_config`은 `temperature <= 0.0`으로 분기한다. 샘플링 분기는 DRY 필드 다섯 개를 모두 전달했지만, greedy 분기는 네 개만 전달하고 `dry_sequence_breakers`는 `SamplingConfig::greedy()`로 흘려보냈다. 그쪽은 `Vec::new()`를 넣는다. DRY는 temperature로 게이팅되지 않으므로, `temperature: 0`에 `dry_multiplier`가 양수인 서버 요청은 breaker 없이 페널티를 적용했다.

프로덕션 코드는 한 줄, 나머지는 이 결함을 계약처럼 고정하고 있던 테스트 단언 두 개다. 그중 `request_options_tests.rs`의 단언은 이슈에 언급되지 않았고, 인접 모듈을 돌리다 발견했다. 수정 후 이 테스트가 실패했다는 사실이 헬퍼 경계가 아니라 서버 경로 전체에서 결함을 확인해 준 근거다.

---

## 1. 문제 정의

### 1.1 배경

`src/execution/sampling.rs`는 CLI와 서버가 요청 필드를 greedy/샘플링 생성에 어떻게 매핑할지 합의하는 유일한 지점이다. `build_sampling_config` 안에는 분기마다 구조체 리터럴이 하나씩 있고, 각 리터럴은 전달하고 싶은 필드를 직접 나열해야 한다. 나열하지 않은 필드는 `..SamplingConfig::greedy()` 또는 명시적 tail로 떨어진다. 이 형태에서는 누락이 보이지 않는다. "의도적으로 초기화한 것"과 "빠뜨린 것"을 타입 시스템이 구분해 주지 않기 때문이다.

greedy 분기에는 바로 이 위험에 대한 주석이 이미 있었다. `xtc_probability`와 `xtc_threshold`는 XTC가 "temperature와 무관하게 적용되는 logits 전처리 단계(위쪽의 repetition/DRY/frequency/presence 페널티와 같은 부류)"라는 설명과 함께 전달된다. 그 주석이 DRY를 같은 부류로 지목하고 있다. DRY 스칼라 필드 네 개는 그에 맞게 전달됐다. 다섯 번째만 그렇지 않았다.

### 1.2 기존 문제점

- **누락은 DRY를 끄는 것이 아니라 세게 만든다.** `dry_sequence_breakers`는 무언가를 켜는 스위치가 아니다. `apply_dry_penalty`(`src/lib/mlxcel-core/src/sampling.rs:846`, `if config.dry_sequence_breakers.contains(&window[p1]) { break; }`)에서 역방향 매칭을 끝내는 종료 조건이다. 벡터가 비어 있으면 루프가 경계 토큰에서 끊기지 않으므로 `match_len`이 호출자가 의도한 창을 넘어 자랄 수 있고, `dry_multiplier * dry_base.powi(match_len - dry_allowed_length)`는 그 초과 길이에 대해 지수적으로 커진다. 루프에서 breaker 검사가 동등성 검사보다 앞에 있으므로, breaker를 없애면 매칭 길이는 늘어나거나 그대로일 뿐 줄지 않는다. 즉 페널티는 요청보다 작아지지 않으며, 경계 너머 토큰까지 마침 일치할 때 실제로 더 커진다. 조용히 아무 일도 안 하는 기능이 아니라, 설정한 값 이상의 페널티가 걸리는 실패 모드다.
- **요청은 정상으로 받아들여졌다.** `dry_sequence_breakers`는 두 요청 형태(`src/server/types/request.rs:601`, `:978`) 모두에 문서화된 요청 단위 필드이고, `routes/chat.rs:1198`과 `routes/native_completion.rs:285`를 거쳐 `src/server/request_options.rs:239`에 도달한다. 거부도, 경고도, 로그도 없었다. `{"temperature": 0, "dry_multiplier": 0.8, "dry_sequence_breakers": [198]}`는 200을 돌려주면서, 호출자가 요청한 적 없는 세기의 페널티가 반영된 출력을 냈다.
- **테스트 두 개가 결함을 계약으로 고정하고 있었다.** `src/execution/sampling_tests.rs:70`은 `temperature: 0`에서 `Vec::<i32>::new()`를 단언했다. 다섯 줄 아래 XTC 단언과 달리 초기화를 정당화하는 주석이 없었고, 같은 테스트가 정작 그 분기가 전달하는 DRY 필드 네 개는 하나도 단언하지 않았다. 이 비대칭이 의도가 아니라 전사(transcription)임을 드러낸다. 의도적 계약이었다면 그렇게 적었을 것이고, 필드 하나가 아니라 분기 전체를 기술했을 것이다.
- **문제가 되는 작업에서 `temperature: 0`이 오히려 일반적이다.** DRY는 반복 루프와 싸우는 운영자가 꺼내 드는 기능이고, 반복 루프 디버깅은 샘플링을 변수에서 빼기 위해 보통 greedy에서 한다. 필드를 떨어뜨린 그 한 분기가, 그런 사용자가 있을 확률이 가장 높은 분기였다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| `temperature: 0`의 출력이 요청 의도와 달라지고, 그 차이가 모델이나 프롬프트 탓으로 돌려짐 | Medium | Medium |
| `temperature > 0`에서 튜닝한 DRY 파라미터가 같은 요청을 `temperature: 0`으로 재생할 때 다르게 동작 | Medium | Medium |
| 이후 읽는 사람이 고정된 테스트 단언을 "DRY는 greedy에서 무력하다"는 의도적 결정으로 읽고 확산 | Low | Low |

---

## 2. 기술적 검토 사항

### 2.1 동작 변화의 범위

DRY가 이미 돌고 있지 않으면 이 변경은 무해하다. `apply_dry_penalty`는 `config.dry_multiplier > 0.0 && !token_history.is_empty()`일 때만 호출되고(`src/lib/mlxcel-core/src/sampling.rs:389`), `default_dry_multiplier` 기본값은 `0.0`이다(`src/server/config.rs:560`). DRY를 켜지 않은 요청은 변경 전후로 동일한 `SamplingConfig`를 본다. DRY를 켜되 breaker를 안 준 요청도 마찬가지다. `unwrap_or_default()`로 어느 쪽이든 빈 벡터가 되기 때문이다. 둘 다 설정한 요청만 달라지고, 그 방향은 요청한 대로다.

### 2.2 greedy 결정성은 그대로

명시적으로 짚을 만한 우려는 greedy 분기에 필드를 추가해서 greedy가 비결정적이 될 수 있느냐다. 그럴 수 없다. `SamplingConfig::greedy()`가 `top_k: 1`, `top_p: 1.0`을 공급하고, 이번 변경은 둘 중 어느 것도 건드리지 않으며, DRY는 선택 이전에 도는 logits 전처리 단계다. 회귀 테스트는 breaker와 함께 `top_k`, `top_p`를 다시 단언한다. 이후 누군가 breaker를 살리려고 greedy 계약을 약화시키는 방향으로 고치는 것을 막기 위해서다.

### 2.3 진짜 확인은 두 번째 테스트에서 나왔다

이슈 #1102는 테스트 하나(`sampling_tests.rs:70`)를 지목했다. 인접 모듈을 돌리자 두 번째가 드러났다. `build_server_generate_options_applies_request_overrides`(`src/server/request_options_tests.rs:129`)는 `temperature: Some(0.0)`, `dry_multiplier: Some(0.9)`, `dry_sequence_breakers: Some(vec![1, 2])`를 구성한다. 버그 리포트의 요청 형태 그 자체인데, 172행에서 `Vec::<i32>::new()`를 단언하고 있었다. 수정 후 `left: [1, 2], right: []`로 실패했다.

이슈가 요구한 판단에 이 사실이 결정적이다. 이슈는 누락이냐, "DRY는 greedy에서 무력"이라는 의도적 결정이냐 두 해석을 제시하고 어느 쪽인지 기록하라고 했다. 의도적 결정이었다면 헬퍼 한 곳에 기록됐을 것이고, 두 계층의 단언은 하나가 다른 하나의 근거이므로 자연히 일치했을 것이다. 실제로는 설명 없는 같은 단언이 두 계층에 독립적으로 나타난다. 의도에서 파생된 것이 아니라 분기에서 베껴진 모양이다. DRY를 temperature 무관 부류로 지목한 greedy 분기 자신의 주석까지 놓고 보면, 자기모순이 없는 해석은 누락 쪽뿐이다.

### 2.4 손대지 않은 인접 경로

`src/lib/mlxcel-xla/src/sampler.rs:218`은 자체 `params.dry_sequence_breakers`로 같은 breaker 종료 매칭을 돌리는데, 이 값은 `src/server/batch/xla_worker_admission.rs:593`에서 이미 만들어진 `sampling.dry_sequence_breakers`를 클론해 채운다. 따라서 별도 수정 없이 이번 수정을 물려받는다. `src/commands/generate.rs:922`는 CLI 경로에서 `Vec::new()`를 하드코딩하고 그것이 범위 결정임을 주석으로 남겨 두었다(#1118). 이번에는 손대지 않았다.

---

## 3. 기술적 선택과 그 이유

### 3.1 나머지 네 개를 떨어뜨리는 대신 필드를 전달한다

| 옵션 | 장점 | 단점 |
|---|---|---|
| **선택: greedy 분기에서 `dry_sequence_breakers` 전달** | 이미 받아들이는 요청이 문서대로 동작; 그 분기 자신의 XTC 주석과 일관; DRY를 켜지 않으면 무해 | 기존 `temperature: 0` + `dry_multiplier > 0` + breaker 호출자의 출력이 달라짐. 이것이 의도한 교정이다 |
| greedy 분기에서 DRY 다섯 개 전부 제거(DRY를 greedy 무력화) | 내부적으로 일관; 규칙이 하나 | DRY를 켠 모든 기존 `temperature: 0` 호출자에게서 페널티를 조용히 끔. 아무도 원했다는 근거 없이 훨씬 큰 동작 변화; 배수만으로 DRY를 게이팅하는 `sampling.rs:389`와 모순 |
| `temperature <= 0`일 때 요청 계층에서 필드 거부 | 조용히가 아니라 크게 실패 | 동작하던 요청이 깨짐; 이 필드는 어떤 temperature에서도 유효; 문서 문제를 400으로 푸는 셈 |

기준은 이슈가 직접 제시했다. 제거 해석은 "`temperature: 0`에 `dry_multiplier > 0`으로 돌리는 사람의 기존 동작을 바꾸므로 더 강한 논거가 필요하다"는 것이다. 트리 안에 그런 논거는 없다. 증거는 반대쪽을 가리킨다.

### 3.2 greedy 테스트에 DRY 계약 전체를 적는다

greedy 테스트는 이제 바뀐 필드 하나가 아니라 DRY 필드 다섯 개를 모두 단언한다. 이유는 결함의 출처 자체에 있다. 필드를 하나씩 적는 구조체 리터럴은 필드가 빠져도 조용히 통과하고, 필드의 일부만 확인하는 테스트는 다음 누락을 잡을 수 없다. 전체 집합을 단언하면 그 분기의 DRY 계약이 한곳에서 읽힌다. 이슈의 세 번째 인수 조건이 요구한 바다.

### 3.3 넓힌 단언에 기대지 않고 이름 붙인 회귀 테스트를 추가한다

`build_sampling_config_keeps_dry_sequence_breakers_at_zero_temperature`는 리포트된 요청(`dry_multiplier: 0.8`, breaker `[198]`)을 독립된 테스트로 다시 적는다. 넓힌 greedy 기본값 단언으로도 회귀는 잡히지만, 그 테스트 이름은 greedy 기본값을 설명하므로 실패 메시지가 다음 사람에게 무엇이 깨졌는지 알려주지 못한다. 이름 붙인 테스트는 버그 리포트를 실패 메시지 안까지 실어 나른다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 3 |
| 추가된 라인 | +41 |
| 삭제된 라인 | -2 |
| 변경된 프로덕션 라인 | 1 (주석 6줄 별도) |
| 테스트 추가 | 1 |
| 교정된 테스트 단언 | 2 |

### 영역별 변경

| 영역 | 파일 | 내용 |
|---|---|---|
| 샘플링 정책 | `src/execution/sampling.rs` | greedy 분기가 `dry_sequence_breakers`를 전달; breaker가 DRY 매칭 종료 조건이므로 누락하면 페널티를 끄는 게 아니라 부풀린다는 점을 주석으로 기록 |
| 단위 테스트 | `src/execution/sampling_tests.rs` | greedy 테스트가 DRY 필드 다섯 개를 계약 설명 주석과 함께 단언; 리포트된 요청에 대한 이름 붙인 회귀 테스트 추가 |
| 단위 테스트 | `src/server/request_options_tests.rs` | `[1, 2]`를 설정한 요청에 대해 빈 breaker 벡터를 단언하던 종단 오버라이드 테스트를 greedy 분기를 지목하는 주석과 함께 교정 |

### 관련 커밋

| Hash | Type | Message |
|---|---|---|
| `6d8badc7` | fix | fix(server): thread dry_sequence_breakers through the greedy branch |

---

## 5. 검증과 후속

### 통과

- `cargo test --profile test-fast --features cuda --lib execution::sampling`: 3 passed.
- `cargo test --profile test-fast --features cuda --lib server::request_options`: 35 passed.
- `cargo fmt --all -- --check`: clean.
- 수정 전 재현 확인: 손대지 않은 단언에 대해 `build_server_generate_options_applies_request_overrides`가 `left: [1, 2], right: []`로 실패. 대상 헬퍼가 아니라 서버 자신의 옵션 구성 계층에서 관찰된 결함이다.

### 다루지 않은 것

페널티 값 자체가 달라지는지 확인하는 생성 수준 단언은 없다. 그런 테스트는 모델과 토큰 히스토리가 필요하고 시드와 체크포인트에 의존한다. 실제로 깨져 있던 계약은 설정 수준 단언이 고정한다. `mlxcel-xla` 샘플러 경로는 교정된 벡터를 클론으로 물려받으며 별도로 실행하지 않았다.

### 후속

`--dry-sequence-breakers`는 시작 시 파싱된 뒤 샘플러에 전혀 도달하지 않는다(#1103). 같은 공백의 운영자 쪽 절반이다. 이 PR은 요청 단위 값이 greedy 분기를 통과하게 만들고, #1103은 서버 전역 기본값이 존재하게 만든다. 플래그 철자는 두 서버 바이너리 사이에서 갈라져 있다(#1109).
