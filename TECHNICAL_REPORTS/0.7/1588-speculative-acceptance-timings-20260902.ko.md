# 기술 보고서: PR #1588 - feat(server): 요청별 speculative 수용 통계를 응답에 보고

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 구현 및 보안 리뷰 사이클
**상태**: 완료 (유닛/라우트 커버리지 통과. 실제 target/drafter 조합 E2E 확인은 머지 오케스트레이터가 수행, 부록 C 참조)
**언어**: Rust
**위험도**: Low (선택적 응답 필드 추가. 비speculative 디코드 경로는 비용과 wire shape 모두 그대로)

---

## 요약

drafter가 요청을 처리하면 스케줄러는 이미 그 요청의 verify 라운드 수와 제안/수용된 draft 토큰 총계를 계산해 놓고, 세 값 모두를 `tracing` 로그 한 줄에 버렸다. 클라이언트는 자기 요청에서 speculation이 도움이 됐는지 알 방법이 없었다. `--draft-max` / `--draft-block-size`를 튜닝하는 운영자와 speculative/classic 경로를 A/B 하는 클라이언트 모두 정확히 그 값을 필요로 한다. PR #1588은 이 카운터를 `GenerationResult`에 실어, native `/completion`의 `timings` 객체에 llama-server가 쓰는 이름 그대로 `draft_n` / `draft_n_accepted`를 얹고 mlxcel 확장 키 두 개(`draft_rounds`, `draft_kind`)를 나란히 둔다. `/v1/chat/completions`도 같은 객체를 최상위 `timings`로 갖는데, b10621이 자신의 OpenAI chat 응답에 `timings` 블록을 얹는 자리가 바로 거기다. 이 블록은 verify 라운드가 실제로 돌았을 때만 나타난다. "drafter가 없음"과 "drafter가 하나도 수용 못 함"이 0의 나열로 뭉개지지 않고 구분된 채로 남는다.

---

## 1. 문제 정의

### 1.1 배경

`mlxcel-server`에서 speculative 요청의 라운드 루프 전체를 소유하는 경로는 셋이고, 셋 다 그 요청 하나에 대한 정확한 수용 요약을 손에 쥔 채 끝난다.

- DFlash B=1 burst. `DFlashGenerator::run`이 `rounds`, `proposed_tokens`, `accepted_tokens`를 담은 `DFlashDiagnostics`를 반환한다 (`src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs`).
- MTP B=1 burst. 제너레이터가 `rounds`, `proposed_tokens`, `accepted_draft_tokens`를 담은 `MtpAcceptanceSummary`를 반환한다.
- tick 협조적 MTP slice (#734). 세션이 요청이 점유하는 모든 스케줄러 tick에 걸쳐 살아 있으므로 `finish_session`이 이미 누적된 총계를 반환한다.

셋 다 이 요약을 내부 용도 두 가지에 쓰고 버렸다. DFlash 쪽은 `info` 레벨로 로깅하고 Prometheus `spec_decode_*` 집계 카운터에 넣었고, MTP 쪽은 적응형 정책(#333)용 `MtpBurstProfile`로 변환했다. 모든 응답 형태가 만들어지는 근원인 `GenerationResult`에는 speculative 필드 자체가 없었으므로, `GenerateEvent::Done`이 HTTP 계층에 도달할 때 수용 정보는 이미 사라진 뒤였다.

### 1.2 그래서 클라이언트가 잃은 것

`/metrics`가 노출하는 것은 MTP 정책의 집계 상태뿐이다. 이것은 "이 배포 전체에서 speculation이 값을 하는가"에는 답하지만 "내 요청에서 값을 했는가"에는 답하지 못한다. 두 질문은 정확히 중요한 지점에서 갈라진다. 수용률은 배포 설정보다 프롬프트의 엔트로피에 훨씬 크게 좌우되므로, 자기 워크로드에서 drafter를 계속 켤지 결정하려는 클라이언트에게 배포 단위 평균은 잘못된 숫자다. 두 경로를 A/B 하던 클라이언트는 벽시계 시간만 비교할 수 있었고, 차이가 수용률 때문인지 부하 때문인지 귀속시킬 수단이 없었다.

### 1.3 이슈 본문의 파일 지도는 이미 어긋나 있었다

이슈는 확장할 timings 타입으로 `src/server/types/response.rs:552`의 `TimingInfo`를, 카운터를 통과시킬 지점으로 `finish_with_cache` 호출부 세 곳을 지목했다. 구현 시점의 트리와 둘 다 맞지 않았다. `response.rs`에 `TimingInfo`는 없고 native timings 타입은 `src/server/types/native_completion.rs`의 `NativeTimings`이며, 이슈가 지목한 세 finalize 지점은 그동안 `SequenceInfo::take_generation_result`(`src/server/batch/sequence.rs`) 한 메서드로 합류해 있었다. 두 번째 어긋남이 더 쓸모 있는 쪽인데, 세 지점을 관통시키는 작업이 파라미터 하나 추가로 줄어들기 때문이다.

### 1.4 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 배치 디코드 핫패스에 토큰당 할당이나 clone이 추가됨 | 발생 시 High | 구조적으로 제거됨. 카운터는 `Copy`이고 요청당 finalize에서 한 번만 계산되며 classic 경로는 `None`을 넘김 |
| 비speculative 응답의 형태가 바뀌어 기존 클라이언트가 깨짐 | 발생 시 High | 제거됨. `None`은 키 자체가 직렬화되지 않고, native/chat 양쪽 형태에 대해 테스트로 고정 |
| 스트림 중간 프레임이 총계처럼 읽히는 진행 중 `draft_n`을 보고함 | Medium | 회피됨. `finish_reason`(chat) 또는 `stop: true`(native)를 이미 실은 프레임만 블록을 실음 |
| drafter가 없는 요청에 0을 보고해 "꺼짐"이 "켰는데 무용지물"처럼 보임 | Medium | 회피됨. `SpeculativeStats::from_counts`가 라운드 1 미만이면 `None` 반환 |

---

## 2. 기술적 검토 사항

**보안.** 새로 파싱하는 입력도, 추가된 요청 필드도 없다. 응답 쪽 write-only 변경이다. 카운터는 서버 자신의 라운드 루프에서 나온 정수이고 요청을 식별할 만한 정보를 담지 않으며, 클라이언트가 이미 제어하는 샘플링 파라미터 이상으로 영향을 줄 수 없다. 짚어볼 만한 정보 노출 지점은 `draft_kind`가 배포 설정을 흘리는가인데, 이 필드는 고정된 세 drafter 계열 중 하나를 가리키고 `GET /props`가 이미 `speculative` 아래에서 draft 모델 basename, kind override, `n_max`를 보고하고 있으므로 그 엔드포인트가 노출하지 않던 것을 노출하지는 않는다.

**성능.** 지켜야 할 쪽은 비speculative 경로다. `--draft-model` 없이 뜬 배포에서는 모든 요청이 그 경로를 탄다. `SpeculativeStats`는 머신 워드 4개 크기의 `Copy` 타입이고, `take_generation_result`에 늘어난 것은 classic 호출부가 `None`으로 넘기는 `Option` 파라미터 하나뿐이라 `None`을 구조체 필드로 옮기는 것이 전부다. 토큰 단위 경로에 추가되는 할당도 clone도 분기도 없다. 값은 finalize에서 한 번, 어차피 요약을 손에 쥐고 있던 코드가 만든다. speculative 경로들도 정책 프로파일용으로 이미 읽던 `Copy` 요약을 읽으므로 추가 작업이 없다.

토큰 단위로 측정 가능한 차이가 하나 있고, 얼버무리기보다 짚어 두는 편이 낫다. `GenerateEvent::Done`은 `GenerationResult`를 박싱 없이 담으므로 enum의 크기가 그 arm으로 정해지고, 응답 채널로 흐르는 모든 `GenerateEvent::Token`이 이제 32바이트를 더 실어 나른다. 기존 enum 크기의 10분의 1 정도이고, 수백 마이크로초 단위인 디코드 스텝에 견주면 무시할 수 있다. `Done` arm을 박싱하면 이 증가분도 사라지고 원래 있던 크기도 줄지만, 그것은 더 크고 별개인 변경이다(후속 조치 참조).

**정확성.** 커버리지 주장을 검증 가능하게 만드는 것이 이 단일 합류 지점이다. MLX 스케줄러의 모든 응답은 `take_generation_result`가 만들고, 따라서 파라미터는 다섯 호출부 각각에서 값이 주어지거나 명시적으로 `None`이며, 어떤 경로가 조용히 빠져나갈 제3의 선택지가 없다.

---

## 3. 기술적 선택과 그 이유

### 3.1 `SequenceInfo`의 필드가 아니라 `take_generation_result`의 파라미터

이슈의 계획은 새 `finish_with_cache_and_speculative`를 finalize 세 지점에 통과시키는 것이었다. `take_generation_result`가 단일 합류점임을 확인한 뒤 남은 유력한 대안은 `SequenceInfo`의 필드였다. 라운드 루프가 값을 써 두고 finalize에서 읽는 방식이다.

파라미터가 두 가지 이유로 이겼다. 첫째, `SequenceInfo`는 프로덕션과 테스트 코드를 합쳐 15곳에서 구조체 리터럴로 만들어지므로, 필드를 추가하면 그중 세 곳만 값을 설정할 수 있는데도 15곳을 고쳐야 하고 테스트 파일이 늘 때마다 16번째가 생긴다. 둘째로 더 중요한 이유는, 필드가 곧 *요청이 들고 다니는 가변 상태*라는 점이다. 나중에 어떤 경로가 값을 일찍 쓰고 다른 경로가 요청의 형태가 바뀐 뒤에 읽는 실수를 부른다. 파라미터는 결과를 만드는 그 순간에, 라운드 루프를 방금 끝낸 코드만 넘길 수 있다. 다섯 호출부는 `speculative_burst.rs`(`finalize_burst_stream`을 통해 두 burst와 slice), `scheduler/decode_tick.rs`, `scheduler/prefill.rs`, 그리고 테스트 두 곳이다.

### 3.2 0으로 채운 구조체가 아니라 라운드 수로 게이팅한 `Option`

`SpeculativeStats::from_counts`는 `draft_rounds == 0`이면 `None`을 반환한다. prefill 안에서 끝난 요청(즉시 EOS, 또는 `n_predict: 1`)은 drafter에게 라운드를 한 번도 주지 않았고, 그것을 `{"draft_n": 0, "draft_n_accepted": 0}`으로 보고하면 클라이언트에게 drafter가 돌았는데 아무것도 제안하지 못했다고 말하는 셈이 된다. 이 블록이 존재하는 이유가 바로 그 구분이므로, 게이트는 어긋날 수 있는 세 호출부가 아니라 생성자에 둔다.

MTP 쪽에는 적응형 정책용으로 `rounds > 0 || probe_rounds > 0`이라는 비슷한 필터가 이미 있었다. 클라이언트용 필터는 의도적으로 더 좁다. classic-step probe 라운드(#736)만으로 이뤄진 실행은 draft를 한 번도 하지 않았고, 정책 입장에서는 정당한 타이밍 샘플이더라도 그것을 speculative 요청이라고 서술하면 사실과 다르다.

### 3.3 모델에는 `DrafterKind`, wire에는 `&'static str`

이슈는 `draft_kind: String`을 명시했다. 대신 `DrafterKind`를 담으면 값이 스케줄러를 통과하는 내내 `Copy`이고 할당이 없으며, 문자열화는 wire 타입으로 미뤄진다. 거기서 `DrafterKind::as_str`이 `&'static str`을 반환하므로 경로 전체가 할당 없이 끝난다. 세 가지로 다르게 적힐 수 있는 문자열 대신 타입 검사를 받는 값이 되고, 정규 이름이 사는 곳이 `DrafterKind::as_str` 한 군데로 유지되어 `--draft-kind`와 그대로 맞는다.

### 3.4 upstream의 이름과 게이트, 그리고 divergence로 선언한 두 개의 추가

`draft_n`과 `draft_n_accepted`는 llama-server 자신의 선택적 쌍이다. upstream의 `result_timings`는 `draft_n > 0`일 때만 이 둘을 `timings`에 덧붙인다. 이름과 게이트를 함께 재현해야 이미 llama-server timings를 읽는 클라이언트가 mlxcel의 것을 고치지 않고 읽는다. 호환 표면이 존재하는 이유가 그것이다.

mlxcel 키 두 개는 취향이 아니라 근거가 필요했다. `draft_rounds`는 upstream 쌍에서 복원할 수 없고, 이것이 없으면 라운드당 평균 수용 길이 `(draft_n_accepted + draft_rounds) / draft_rounds`를 계산할 수 없는데 블록 크기를 튜닝하는 운영자가 실제로 원하는 숫자가 그것이다. `draft_kind`는 upstream에 대응물이 없다. upstream은 draft 모델 개념이 하나인데 mlxcel에는 drafter가 셋이기 때문이다. 둘 다 upstream이 이미 선택적으로 두고 있는 객체에 더해지는 키이므로 b10621 클라이언트가 읽는 것은 아무것도 바뀌지 않는다.

에픽 #1431의 규칙에 따라 이 차이는 `notes`의 자유 서술이 아니라 `compat/llama-server/b10621/routes.json`의 두 native 라우트 항목에 검사되는 `divergence` 항목으로, 근거와 재검토 조건을 붙여 기록했다. 같은 편집에서 항목 산문의 영구 차이 개수를 둘에서 셋으로 고쳐, 기계가 검사하는 배열과 그 옆의 산문이 어긋날 수 없게 했다.

### 3.5 chat의 `timings`는 draft 절반이 아니라 upstream의 블록 전체다

llama-server는 자신의 OpenAI chat 응답에도 `timings` 객체를 얹고, 그 객체는 native 라우트가 쓰는 `result_timings`와 같은 것에서 만들어진다. 필드 이름을 mlxcel 고유어가 아니라 `timings`로 정한 이유다.

첫 구현은 `draft_*` 키 네 개만 담았다. 객체 전체가 speculation 활성 여부로 게이팅되므로 아홉 개 기본 키까지 drafter를 따라 나타났다 사라진다는 것이 근거였다. 리뷰가 그 형태의 더 날카로운 귀결을 짚었다. `if (res.timings) show(res.timings.predicted_per_second)`처럼 쓰인 b10621 클라이언트는 이전에는 항상 부재 분기를 탔는데, 이제 존재 분기를 타고 `undefined`를 읽는다. upstream이 이미 정의해 둔 키 아래의 부분 객체는 키가 아예 없는 것보다 나쁘고, 깜빡임 논거는 두 형태를 구별하지 못한다. 어느 쪽이든 존재 여부는 게이팅되기 때문이다.

그래서 이 블록은 upstream의 객체 전체, `cache_n`부터 `predicted_per_second`까지에 flatten된 draft 절반을 더한 것이다. 이것을 만드는 `chat_timings` 함수 하나가 존재 규칙도 함께 소유한다. 남은 차이는 내용이 아니라 존재다. upstream은 모든 chat completion에 블록을 얹고 mlxcel은 drafter가 처리한 요청에만 얹는데, 무조건 얹으면 모든 비speculative 배포의 응답 형태가 바뀌기 때문이다. 그것은 수용 통계를 추가한 부수 효과가 아니라 자체 파급 범위를 가진 chat 라우트 호환성 결정이다.

매니페스트의 두 chat 라우트 항목은 `divergence`가 빈 `supported`에서, 그 차이와 근거와 재검토 조건을 담은 `by_design`으로 옮겼다. 두 항목은 `timings`를 전혀 내보내지 않으면서 `supported`를 주장하고 있었으니 이 변경 이전부터 이미 틀린 주장이었다. #1314이 그 주장을 사실로 만든 이슈이고, 그래서 지금 두 항목에 기록된 이슈이자 `pin.json`의 routes shard 소유자 목록에 추가된 번호이기도 하다.

### 3.6 `NativeTimings`에 `Option` 필드 네 개가 아니라 flatten

`NativeTimings`는 `#[serde(flatten)] Option<SpeculativeTimings>`를 담고, `with_speculative` 빌더로 붙인다. flatten은 draft 키들을 `timings` 객체 위에 바로 놓는데 upstream이 자기 쌍을 두는 자리가 거기이고, 동시에 chat 라우트가 통째로 재사용하는 타입 하나를 유지한다. 빌더 덕분에 `NativeTimings::new`의 시그니처는 그대로다. 이 함수의 산술은 고정된 b10621 바이너리를 상대로 측정한 것이고 다섯 파라미터가 기존 테스트로 고정되어 있어, 늘렸다면 얻는 것 없이 `native_completion_tests.rs`까지 파급됐을 것이다.

---

## 4. 구현 상세

### 4.1 카운터와 단일 합류점

`SpeculativeStats`(`src/server/model_provider.rs`)는 `draft_kind: DrafterKind`, `draft_rounds`, `draft_n`, `draft_n_accepted`를 담고, `GenerationResult`에 `speculative: Option<SpeculativeStats>`가 추가된다. `SequenceInfo::take_generation_result`는 이 블록을 세 번째 파라미터로 받아 결과에 대입한다.

### 4.2 값을 생산하는 세 경로

| 경로 | 카운터 출처 | 블록을 만드는 지점 |
|---|---|---|
| DFlash B=1 burst | `DFlashDiagnostics { rounds, proposed_tokens, accepted_tokens }` | `run_dflash_on_qwen35`, 기존 `DFlash diagnostics` 로그 라인 옆 |
| MTP B=1 burst | `MtpAcceptanceSummary { rounds, proposed_tokens, accepted_draft_tokens }` | `run_mtp_burst`, 적응형 정책이 받는 `MtpBurstProfile` 옆 |
| tick 협조적 MTP slice | 같은 요약, `generator.finish_session`에서 | `BatchScheduler::finalize_speculative_slice` |

두 요약 타입 모두 `Copy`이므로 각 지점은 정책 경로가 읽는 값을 그대로 읽고 다시 계산하지 않는다. slice 경로에는 slice 간 누적이 필요 없다. 세션이 요청이 점유하는 모든 tick보다 오래 살아서 `finish_session`이 이미 총계를 반환한다. 이 경로는 kind를 가정하지 않고 `SpeculativeDispatch::drafter_kind()`로 확인하며, `Option::zip` 덕분에 drafter를 돌리지 않는 dispatch는 기본값 대신 블록 없음을 만든다.

DFlash 드라이버의 반환 타입은 3원소 튜플에서 이름 있는 `DFlashTargetRun` 구조체로 바꿨다. 나중에 읽는 사람이 위치를 세어 가며 식별해야 하는 네 번째 튜플 원소 없이 진단값이 함수 밖으로 나올 수 있게 하기 위해서다.

### 4.3 의도적으로 아무것도 보고하지 않는 경로

기본 비활성인 B>1 배치 burst는 `None`을 넘긴다. 이 경로의 라운드 루프는 행별 토큰만 반환하고 행별 수용 카운터를 내지 않으며, 윈도우 전체 값을 한 행에 귀속시키면 틀린 값이 된다. 분리형(disaggregated) 라우터 프런트도 아무것도 보고하지 않는다. 이쪽은 완성된 `GenerationResult`가 아니라 prefill/decode 핸드오프 프로토콜로 스트림을 조립하는데, 그 프로토콜은 생성 토큰 수만 실어 나르고 수용 카운터를 담지 않는다. 거기에 블록을 노출하는 것은 응답 성형이 아니라 핸드오프 프로토콜 변경이다. 두 경우 모두 읽는 사람이 볼 자리, 즉 코드와 `docs/speculative-acceptance.md`에 남겼다.

### 4.4 wire

`SpeculativeTimings`(`src/server/types/native_completion.rs`)는 `draft_n`, `draft_n_accepted`, `draft_rounds`, `draft_kind` 순으로, upstream 쌍을 앞에 두고 직렬화하며, `NativeTimings`의 아홉 기본 키 뒤에 flatten된다. `ChatCompletionResponse`와 `ChatCompletionChunk`는 `#[serde(skip_serializing_if = "Option::is_none")] timings: Option<NativeTimings>` 필드와, 기존 `with_cached_tokens` 스타일의 `with_timings` 빌더를 갖는다. 값은 `chat_timings`가 채우는데, 이 함수 하나가 두 형태 모두에 대해 "drafter가 처리한 요청에만"이라는 규칙을 소유한다.

라우트 배선은 이렇다. `NativeOutcome`에 `speculative` 멤버가 생겨서, native 비스트리밍 본문과 스트리밍 최종 프레임이 원래 공유하던 `build_native_response` 한 함수를 통해 만들어지고 서로 어긋날 수 없다. 토큰별 `timings_per_token` 프레임은 손대지 않아 기본 아홉 키만 담는다. chat 라우트에서는 비스트리밍 반환 네 지점과 스트림의 finish 청크에 블록을 붙인다.

---

## 5. 학습 포인트

### 5.1 키의 부재와 0은 서로 다른 대답이고, 정직한 쪽은 하나뿐이다

이런 변경에서 솔깃한 선택은 필드를 필수로 만들고 drafter가 없으면 0을 보고하는 것이다. 모든 소비자 코드가 단순해지기 때문이다. 동시에 이 숫자들이 존재하는 유일한 구분도 사라진다. `draft_n_accepted: 0`을 본 클라이언트는 speculation이 꺼져 있는지 켜져 있는데 쓸모없는지 구별할 수 없는데, 두 경우의 조치는 정반대다. 호출부가 아니라 생성자에서 게이팅하는 선택이, 값을 생산하는 경로가 늘어나도 이 규칙이 어긋나지 않게 한다.

### 5.2 호환성은 이름 *과* 게이트다

llama-server의 `draft_n` 철자를 맞추면서 무조건 내보냈다면 엄격한 클라이언트는 여전히 깨졌을 것이다. upstream의 이 키는 라운드 1 미만에서 아예 없고, 클라이언트는 키의 존재 자체를 신호로 다룰 수 있기 때문이다. 게이트 재현은 이름 재현만큼 핵심이었다.

### 5.3 생태계가 이미 정의한 키 아래의 부분 객체는 키가 없는 것보다 나쁘다

chat 블록의 첫 구현은 `draft_*` 키 네 개만 담았고, 그것이 보수적으로 보였다. 키가 적고 표면이 작고 배포 설정을 따라 깜빡이는 비율 값도 없다. 실제로는 반대였다. `timings`는 b10621 클라이언트가 이미 존재를 확인하는 키이고, 그런 클라이언트는 하나같이 기본 키 중 하나를 그 위에서 읽는다. 기본 키 없이 그 키만 내보내면 멀쩡하던 부재 분기가 살아 있는 `undefined`로 바뀐다. 교훈은 이 필드를 넘어 일반화된다. 다른 구현이 이미 정의한 키를 추가할 때는 그들이 채우는 대로 채우거나 다른 이름을 고르는 수밖에 없다. 절반만 채우는 것이 이전까지 멀쩡했던 클라이언트를 깨뜨리는 유일한 선택지다.

### 5.4 뒤늦게 찾은 합류점이 그것을 대체하는 계획보다 값지다

이슈의 계획은 `take_generation_result`를 단일 finalize 합류점으로 만든 리팩터링보다 앞선 것이라, 그대로 따랐다면 두 번째 `finish_with_cache` 변형과 세 곳의 관통 작업이 나왔을 것이다. 트리를 먼저 확인하니 파라미터 하나에 호출부 다섯 곳이 됐고, 그중 셋은 단어 하나 수정이다. 일반화하면 이렇다. 이슈의 구현 계획이 병렬적인 호출부 여러 곳을 지목하면, 무엇을 관통시키기 전에 그것들이 그동안 하나로 합쳐지지 않았는지 확인한다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

- **draft / verify 라운드**: `k`개 토큰을 제안하는 drafter forward 한 번과 그것을 검증하는 target forward 한 번. `draft_rounds`가 세는 단위.
- **수용(acceptance)**: 한 라운드의 제안 중 target이 남긴 개수. `draft_n_accepted / draft_n`은 draft 작업 중 살아남은 비율이고, `(draft_n_accepted + draft_rounds) / draft_rounds`는 target forward 하나당 나온 토큰 수로 처리량을 좌우한다.
- **bonus 토큰**: 제안이 몇 개 수용됐는지와 무관하게 각 라운드가 target 자신의 verify 로짓에서 내보내는 토큰. 라운드당 평균 수용 길이를 구할 때 나누기 전에 `draft_rounds`를 더하는 이유다.
- **`result_timings`**: llama-server의 timings 구조체. 이번 변경이 재현한 선택적 `draft_n` / `draft_n_accepted` 쌍이 여기 있다.

### 관련 PR/이슈

- #1314: 이번 이슈.
- #1431: llama-server b10621 호환 에픽. divergence를 검사되는 필드로 두는 규칙을 이번 변경이 따랐다.
- #1477 / #1441 / #1466 / #1525: native `/completion` 라우트의 선행 호환 작업. 이번 변경이 그 라우트 항목을 수정했다.
- #734: tick 협조적 MTP slice. 이번 변경이 그 세션 누적에 기댄다.
- #333 / #736: 적응형 MTP 정책과 classic-step probe 라운드. 이번 변경이 클라이언트용 블록에서 그 필터를 의도적으로 좁혔다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 27 |
| 추가 라인 | 약 950 |
| 삭제 라인 | 약 90 |
| 신규 공개 타입 | 2 (`SpeculativeStats`, `SpeculativeTimings`) |
| 신규 테스트 | 12 |

### 카테고리별 변경

- **모델 / 스케줄러**: `src/server/model_provider.rs`, `src/server/model_worker.rs`, `src/server/batch/sequence.rs`, `src/server/batch/speculative_burst.rs`, `src/server/batch/scheduler.rs`, `src/server/batch/scheduler/decode_tick.rs`, `src/server/batch/scheduler/prefill.rs`, `src/server/florence2_worker.rs`.
- **wire 타입**: `src/server/types/native_completion.rs`, `src/server/types/response.rs`, `src/server/types/stream.rs`.
- **라우트**: `src/server/routes/native_completion.rs`, `src/server/routes/chat.rs`, `src/server/router_front.rs`(의도적 미지원을 문서화한 것뿐).
- **테스트**: `src/server/types/native_completion_tests.rs`, `src/server/routes/native_route_tests.rs`, `src/server/batch/speculative_slice_tests.rs`, `src/server/model_provider_test_support.rs`, 그리고 기존 호출부의 기계적인 `None` 추가.
- **문서 및 매니페스트**: `docs/speculative-acceptance.md`, `docs/llama-server-compat.md`, `compat/llama-server/b10621/routes.json`(native 라우트 두 항목 보완, chat 라우트 두 항목을 `by_design`으로 이동), `compat/llama-server/b10621/pin.json`(routes shard 소유자).

---

## 8. 후속 조치

### 모니터링 필요

- `predicted_n == draft_n_accepted + draft_rounds`(EOS 라운드 기준 오차 1 이내)라는 E2E 항등식은 실제 drafter가 필요해서 유닛 테스트가 아니라 오케스트레이터의 실제 체크포인트 실행에서만 확인된다. 라운드의 bonus 토큰 방출 방식이 바뀌면 유닛 레인에서는 조용히 깨진다.

### 향후 개선 사항

- upstream처럼 모든 chat completion에 `timings`를 내보내면 chat 라우트의 마지막 차이가 닫힌다. 비speculative chat 응답 형태가 바뀌므로 담당자가 필요하고, upstream이 같은 OAI-compat 빌더에서 `/v1/completions`에도 블록을 내보내므로 그쪽을 함께 옮길지도 결정해야 한다.
- `GenerateEvent::Done(Box<GenerationResult>)`로 바꾸면 이번에 토큰당 전송에 더해진 32바이트가 사라지고 원래 있던 약 250바이트도 줄어든다. 이번 범위 밖이고 별도 이슈로 다룰 만하다.
- 분리형 라우터 프런트는 아무것도 보고하지 않는다(4.3). 핸드오프 프로토콜로 카운터를 실어 나르면 마지막 서빙 경로가 닫히지만 프로토콜 필드가 하나 늘어난다.
- B>1 배치 burst는 라운드 루프가 행별 수용 카운터를 내놓아야 무엇이든 보고할 수 있다. 오늘 기준 기본 비활성이므로 급하지 않다.
- 토큰별 draft / verify 시간(`draft_time_ms`, `verify_time_ms`)의 wire 노출은 #1314에서 명시적으로 범위 밖이었다. 진단값에는 이미 둘 다 있으므로 노출할 근거가 생기면 배선은 준비되어 있다.

---

## 부록

### A. 테스트 결과

| 스위트 | 결과 |
|---|---|
| `--lib server::batch::speculative` | 87 passed |
| `--lib server::types` | 98 passed |
| `--lib native_route_tests` | 32 passed |
| `--lib stream_route_tests` | 9 passed |
| `--lib server::routes::chat` | 48 passed |
| `--lib server::batch::scheduler` | 137 passed |
| `--lib server::model_provider` | 57 passed |
| `--lib server::llama_compat_tests` | 3 passed |
| `--lib server::batch::stop_sequence_tests` | 8 passed |
| `--lib server::max_tokens_route_tests` | 7 passed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `cargo check --profile test-fast --features metal,accelerate --bins` | clean |
| `make verify-llama-compat` | 자체 negative 케이스 포함 통과 |

### B. 새 테스트가 고정하는 것

- `timings_carries_no_draft_keys_without_a_drafter`: 아홉 키 그대로, `draft_*` 키 없음.
- `timings_carries_the_b10621_draft_pair_for_a_speculative_request`: 열세 키, upstream 쌍과 확장 두 개, 기본 키들은 원래 값 유지.
- `a_zero_round_speculative_run_reports_no_draft_block`, `every_drafter_kind_renders_its_canonical_name`: 게이트와 세 정규 이름. 오늘 기준 도달 가능한 것은 셋 중 둘이다. `internal-mtp`은 classic dispatch로 해석되고 burst 게이트가 그것을 거절하므로, 그 kind로 speculative 처리되는 요청이 없고 블록을 보고하는 요청도 없다.
- `the_done_result_carries_the_acceptance_counters_of_a_drafted_request`: 실제 2라운드 제너레이터 스크립트를 돌려 `Done` 이벤트가 라운드 루프가 만든 카운터를 싣는지, `draft_rounds > 0`과 `0 < draft_n_accepted <= draft_n`이 성립하는지 확인.
- `the_done_result_reports_no_acceptance_for_an_undrafted_finish`: classic 경로의 계약.
- 라우트 레벨: `/completion`, `/completions`, `/v1/chat/completions`의 네 키와 chat 본문의 열세 키 순서. drafter 없을 때 두 형태 모두에서의 부재. chat 스트림에서 `finish_reason` 청크만 `timings`를 싣는다는 점(스크립트 provider로 실제 콘텐츠 프레임이 있는 스트림을 만들어 부정 절반이 실효를 갖게 함). drafted native 스트림의 `timings_per_token` 부분 프레임이 정확히 아홉 기본 키만 담고 최종 프레임이 열세 개를 담는다는 점.

### C. Merge 이후 검증 (오케스트레이터)

```bash
./target/release/mlxcel-server \
  -m models/mlx/qwen3.5-4b-4bit \
  --draft-model models/mlx/qwen3.5-4b-dflash \
  --draft-kind dflash \
  --draft-block-size 16 \
  --port 8080

curl -s localhost:8080/completion -H 'content-type: application/json' \
  -d '{"prompt":"Write the numbers one to thirty in words.","n_predict":96,"temperature":0}' \
  | jq '.timings | {draft_kind, draft_rounds, draft_n, draft_n_accepted, predicted_n}'

curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.5-4b-4bit","messages":[{"role":"user","content":"Write the numbers one to thirty in words."}],"max_tokens":96,"temperature":0}' \
  | jq '.timings'
```

기대값: `draft_kind == "dflash"`, `draft_rounds > 0`, `0 < draft_n_accepted <= draft_n`, EOS 라운드 기준 오차 1 이내로 `predicted_n == draft_n_accepted + draft_rounds`, 그리고 해당 요청의 서버 `DFlash diagnostics` 로그 라인에 같은 값이 찍힐 것. chat 본문의 `timings`는 그 네 키만 담는다. `--draft-model` 없이 재시작하면 `/completion`의 `timings`에 네 키가 하나도 없고 chat 본문에는 `timings` 키 자체가 없어야 한다.
