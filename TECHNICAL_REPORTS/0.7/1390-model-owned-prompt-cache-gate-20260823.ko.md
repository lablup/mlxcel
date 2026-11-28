# 기술 보고서: PR #1390 - 프롬프트 캐시가 K/V 없는 shadow paged 항목을 기부·채택하던 문제 수정

**날짜**: 2026-08-23
**작성자**: 신정규
**상태**: 완료
**언어/기술**: Rust
**위험도**: 수정 전 높음 (기본 서버 설정에서 두 모델 계열의 turn 2 이력이 조용히 소실). 변경 자체는 좁게 한정됨

---

## 요약

기본 서버 설정에서 Gemma 3나 Llama 4 시퀀스는 **shadow block-table 회계 목적으로만** paged backend에 할당되고, 실제 K/V는 모델의 `ModelOwnedSequenceState`에 있다. 프롬프트 캐시는 그것을 몰랐다. shadow block table을 paged 항목으로 떼어 저장했고, 대화를 잇는 다음 요청이 그것을 채택했다. 스케줄러는 `cached=160/190`을 보고하고 그 160토큰의 prefill을 건너뛰었는데, 새 시퀀스의 모델 내부 캐시는 비어 있으므로 모델은 **30토큰 suffix만 보고 디코드**했다.

사용자에게 보이는 증상은 **turn 2가 이력 없이 답하는 것**이고, 에러도 경고도 없다. `gemma3-1b-4bit` 3턴 대화의 두 번째 턴이 "I understand. You're repeating yourself. It's a very frustrating process"를 냈는데, 같은 대화를 `--no-prompt-cache`로 돌리면 정상 응답이 나왔다.

이 유닛은 **수용 테스트가 수정보다 먼저 존재했다**는 점에서 특이하다. 이 배치 앞부분에서 만든 멀티턴 차분 하네스가 이미 두 체크포인트에서 결함을 재현하고 있었으므로, 그 실행이 초록으로 바뀌는 것이 곧 증명이고 단위 테스트로 논증할 필요가 없었다.

## 1. 문제 정의

### 1.1 배경

손상이 성립하려면 세 가지가 맞물려야 했다.

1. **할당.** paged override에서 model-owned 계열은 `model.make_caches()` 자리표시자와 함께 `PagedKvCache` backend를 받는데, Gemma 3와 Llama 4에서 그것은 **빈 벡터**다. 그래도 `pool.append_tokens`가 shadow table용 실제 pool 블록을 할당하므로 `retained_block_count() > 0`, `seq_len() > 0`이 된다.
2. **기부.** `donate_finished_sequence_cache`의 유일한 model-owned 탈출구가 `supports_snapshot_reuse()`였는데 Gemma 3·AFMoE·Llama 4는 전부 `false`를 반환하고, 그다음이 **할당된** backend 검사인데 그 값은 `ModelOwned`가 아니라 `PagedKvCache`다.
3. **공백 판정.** `DetachedKvSet::is_empty`의 paged arm이 `seq_len() == 0 || retained_block_count() == 0`을 봤다. shadow 항목에서는 둘 다 false라 삽입되고 나중에 채택됐다.

### 1.2 기존 문제

이 실패는 **설계상 조용하다.** 아무것도 던지지 않고, 로그도 없고, 출력이 유창하다. 그래서 서버가 문맥을 버린 것이 아니라 모델이 대화를 잘 못 따라가는 것처럼 읽힌다.

### 1.3 위험 평가

수정 전 기준 높음. 기본 설정에서, 프로젝트 권장 테스트 세트에 든 두 계열이 대상이다. 변경 자체는 좁다. 리뷰가 확인한 바로, 자연 backend가 `ModelOwned`인 계열이 이 PR 이전에 기부를 하려면 할당된 backend가 `ModelOwned`가 아니어야 하고, 그러려면 paged override가 필요하며, 그러려면 `supports_batching() && supports_paged_decode_backend()`가 필요하다. 그 집합은 정확히 Gemma 3, Llama 4, Qwen 3.5이고 Qwen 3.5는 snapshot 분기에서 먼저 빠진다. **이 게이트가 잘 돌던 캐시를 뺏는 구성은 존재하지 않는다.**

## 2. 변경 요약

12개 파일, 약 710줄 추가.

| 영역 | 변경 |
|---|---|
| `src/server/batch/scheduler.rs` | 모델의 자연 backend를 보는 donate 게이트와 adopt 게이트 |
| `src/lib/mlxcel-core/src/cache/paged_detach.rs` | `detach_paged`가 핸들 없는 paged 시퀀스를 **블록을 고정하기 전에** 거절, `clone_eligible`이 dense 핸들 요구 |
| `src/server/prompt_cache/entry.rs` | `is_empty`가 핸들 0인 paged 집합을 empty로 판정 (`paged_set_is_empty`로 추출) |
| `metrics.rs`, `routes/cache.rs`, `routes/metrics.rs` | `PromptCacheRejectReason::ModelOwnedState` 신설, `reject_model_owned_state`와 `/metrics` 라벨로 노출 |
| `docs/turbo-kv-cache.md` | model-owned 계열이 회계 목적으로만 paged backend에 올라간다는 사실 기록 |
| 테스트 | `scheduler_model_owned_cache_tests.rs`(신규) 외 3개 |

## 3. 기술적 선택과 그 이유

### 3.1 술어는 모델의 자연 backend다. 할당된 것도 override도 아니다

이것이 수정의 전부이고, 셋 중 잘못 고르면 **반대 방향으로 실패한다.** 할당된 backend를 보면 아무것도 안 바뀌고(그게 거짓말하는 값이다), override를 보면 캐시를 계속 써야 할 계열의 캐시를 꺼 버린다.

논증이 아니라 코드를 읽어 정했고, 트리 안에 선례가 둘 있었다.

- `CachePool::allocate_with_layout`이 이미 `model.sequence_state_layout().backend`를 읽어 `natural_backend`라 이름 붙이고, **override가 불일치할 수 있다는 바로 그 이유로** pool-backing 판단에 쓴다.
- `BatchScheduler::handoff_supported`가 #708에서 같은 이유로 같은 사실에 게이트를 건다.

`sequence_state_layout_override`는 검토로 배제했다. 서버 설정(`decode_storage_backend`, `num_layers`, KV 캐시 모드, 블록 크기)만으로 구성되고 모델에게 K/V를 어떻게 다루는지 **묻지 않는다.** 게다가 dense backend에서는 `None`인데, 거기서도 model-owned 계열은 같은 거절이 필요하다.

두 값의 불일치는 이제 주석이 아니라 **테스트로 고정**했다. `paged_override_does_not_change_the_model_owned_natural_backend`가 실제 `Gemma3Wrapper`를 만들어 모델은 `ModelOwned`, override는 `PagedKvCache`를 보고하며 할당된 시퀀스의 레이어별 캐시가 0개임을 단언한다.

### 3.2 방어 심층, 정확하게 기술하기

이 변경은 항 다섯 개를 추가하는데, 각각이 무엇을 사는지 정확히 적을 값어치가 있다. PR 본문 초안이 "이 중 어느 하나만으로도 손상을 막는다"고 **과장했기 때문이다.**

**독립적으로 충분한 것은 셋이다**: donate 게이트, adopt 게이트, 핸들 없는 시퀀스를 거절하는 `detach_paged`.

**둘은 adopt 쪽에서 개별이 아니라 합쳐서 충분하다.** `clone_eligible` 단독으로는 채택을 못 막는다. clone 부적격 paged 집합이 take 경로로 떨어지고 거기서는 `is_empty`가 막는다. 반대로 `is_empty` 단독으로는 clone 경로를 못 막는다. 그 경로는 `is_empty`를 아예 조회하지 않는다.

`detach_paged`는 읽기 전용 borrow 안에서, `self.active.remove` 이전, `retain_block` 루프보다 한참 앞에서 거절한다. 그래서 거절이 pool 예산에 무해하다. 테스트가 active 카운트와 refcount 불변을 둘 다 단언해 이 성질을 읽기에 맡기지 않고 고정한다.

### 3.3 잘못 선언된 layout이 이제 안전한 방향으로 실패한다

기록해 둘 부수 효과: 실제로는 외부 캐시를 쓰면서 `model_owned`라고 잘못 보고하는 계열은 이제 **틀린 답을 내는 대신 프롬프트 캐시를 잃는다.** 코드베이스가 이미 이 선언을 load-bearing으로 취급하고 있어서(`phi4mm_vl.rs`와 `falcon_ocr.rs`가 정확히 그 위험을 설명하는 주석과 함께 명시적 dense override를 달고 있다), 게이트는 새 불변식을 만든 게 아니라 기존 것을 물려받는다.

## 4. 검증

### 4.1 결함을 이미 재현하던 하네스

이 배치 앞부분에서 만든 멀티턴 차분 하네스는 3턴 대화를 캐시 ON과 `--no-prompt-cache`로 각각 돌려 **생성된 모든 토큰을 비교**하고, 분기 지점 자신의 top-2 마진이 jitter 바닥 0.05 아래일 때만 면제한다.

| 체크포인트 | 수정 전 | 수정 후 |
|---|---|---|
| `models/gemma3-1b-4bit` | FAIL, turn 2 step 0, 마진 0.516 | **PASS, 전 턴 동일** |
| `models/gemma-3-4b-it-4bit` | FAIL, turn 3 step 0, 마진 0.203 | **PASS, 전 턴 동일** |
| `models/llama-4-scout-17b-4bit` | 미측정 | **PASS** |
| `models/llama-3.1-8b-4bit` | PASS | PASS, 변화 없음 |
| `models/internlm3-8b-4bit` | PASS, 동점 면제 | PASS, 같은 동점, 같은 마진 0.00000 |

이 결과를 얻으려고 **하네스를 수정하지 않았다.** internlm3의 면제가 변경 전과 같은 지점의 같은 정확한 동점이라는 것이, 비교 기준 자체가 움직이지 않았음을 확인해 주는 대조군이다.

`llama-4-scout-17b-4bit`은 덤이었다. 브리프는 이슈가 지목한 두 번째 계열의 로컬 체크포인트가 없다고 가정했는데, 있었다.

### 4.2 있어야 할 곳의 캐시는 살아 있다

이 수정의 특징적 실패는 **모두의 프롬프트 캐시를 꺼 버리는 것**이라, 가정하지 않고 측정했다. 모델마다 같은 3턴 대화 후 `GET /v1/cache/stats`:

| 모델 | 계열 | inserts | hits | snapshot ins/hit | reject_model_owned_state |
|---|---|---|---|---|---|
| `llama-3.1-8b-4bit` | dense-natural | 3 | 2 | 0 / 0 | 0 |
| `gemma-4-e2b-it-4bit` | snapshot reuse | 0 | 0 | **5 / 2** | 0 |
| `gemma3-1b-4bit` | model-owned | 0 | 0 | 0 / 0 | **3** |
| `llama-4-scout-17b-4bit` | model-owned | 0 | 0 | 0 / 0 | **3** |

서버 로그도 일치한다. Llama 3.1은 여전히 채택하고(`cached=96/133`, `cached=128/165`), Gemma 3는 이제 `cached=0/29`, `0/52`, `0/84`를 남긴다.

여전히 기부하는 snapshot 계열로 Gemma 4를 지목했고, 리뷰가 `supports_snapshot_reuse() == true`인 13개 계열을 독립적으로 열거해 전부 model-owned-natural임을, 따라서 그중 어느 것도 새 게이트에 걸리지 않음을 확인했다.

### 4.3 테스트가 공허하지 않다는 반사실 증명

가드 항 다섯 개를 무력화하고 스위트를 재실행한 뒤 백업에서 복원했다. 가드를 끄면 `model_owned_paged_family_never_donates_or_adopts`가 `a shadow paged sequence must never reach the store; left: 1, right: 0`과 `DetachedPagedCacheSet dropped with 2 retained blocks`로 실패하는데, 이는 mock이 아니라 **실제 스케줄러를 통해 주조된 #1346의 정확한 형태**다.

### 4.4 게이트

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8367 통과, 0 실패. 로컬 CI(이번 세션 GitHub Actions 불가로 `ci.yml` 잡 10개 중 8개): 7 pass, 0 fail, 2 skip. skip 둘은 CUDA 툴체인이 필요한 `OpenXLA feature compile`과 `link`인데, 이 변경은 `xla`·`iree` 경로를 전혀 건드리지 않아 무관하다.

게이트 한 번이 알려진 플레이키 `text_only_forward_produces_finite_logits`(#997)로 실패했다. 상시 절차로 판정했다. 브랜치가 hunyuan 경로가 닿는 파일을 하나도 안 건드리고, 격리에서 3/3 통과하며, 같은 커밋 범위가 앞서 게이트를 통과했고, 재실행이 초록이었다. 이 배치의 발생 데이터를 #997에 코멘트로 남겼다.

## 5. 리뷰에서 나온 지적

MEDIUM 초과 없음. MEDIUM 2건, LOW 6건 중 8건 반영.

**가장 값진 지적은 adopt 게이트에 테스트가 없다는 것이었다.** 가드 5개에 테스트 5개라 깔끔한 매핑처럼 보였지만, 그중 하나는 어떤 가드도 안 덮는 전제 테스트이고, donate/adopt 테스트의 모든 단언은 adopt 게이트를 지워도 초록으로 남는다. donate 게이트가 이미 store를 비워 둬서 건너뛴 조회가 어차피 miss였을 것이기 때문이다. **위의 반사실 검증조차 이걸 못 잡았다.** 나머지 넷에는 유효했으니까. 수정은 `store.stats().lookups == 0` 단언 하나이고, 이것이 "store 조회 전에 리턴했다"와 "조회했고 miss였고 넘어갔다"를 가르며, 아래의 의도적 계측 변화도 산문 대신 테스트로 고정한다.

다른 MEDIUM: `docs/turbo-kv-cache.md`가 AFMoE를 paged backend에 할당되는 계열로 적었다. 아니다. `afmoe.rs`가 `supports_batching()`에서 `false`를 반환하고 `effective_decode_storage_backend`가 그것을 요구하므로, AFMoE는 할당된 backend가 이미 `ModelOwned`인 dense backend에 머문다. 이슈 본문 자체가 여기서 다른 방향으로 틀렸고(`supports_paged_decode_backend()`를 탓했다) 문서가 그 오류를 물려받았다.

그 외 반영: 존재하지 않는 호출자를 적고 있던 `detach_paged`의 `Used by:` 로스터 정정, 관측성 결과 2건 문서화, doc 주석에 들어간 em dash 4개 제거, PR 본문의 과장 표현 2건 정정.

## 6. 의도적 관측성 변화

둘이고, PR 본문이 아니라 `CacheStatsResponse::reject_model_owned_state`에 문서화했다.

- `supports_snapshot_reuse()` 계열은 snapshot miss 시 KV `lookups` 카운터를 더 이상 올리지 않는다. adopt 게이트가 `store.lookup_longest_prefix` 전에 리턴하기 때문이다. 그 조회는 이 계열에서 miss만 날 수 있었으므로 `hit_rate`가 더 정직해지지만, `/v1/cache/stats.lookups`와 Prometheus `mlxcel_prompt_cache_misses_total`이 **설계상 불일치**하게 된다. miss 카운터는 호출자 쪽에서 기록되기 때문이다. 전에는 일치했다.
- Gemma 3와 Llama 4에서는 정상 완료마다 이 거절이 발동하므로 `last_reject_reason`이 `model_owned_state`에 고정되어 `oversized` 같은 더 흥미로운 이전 거절을 가린다. 사유별 카운터는 계속 분리되어 있고, 이 계열에서는 그쪽을 읽어야 한다.

## 7. 검증하지 못한 것

실제 체크포인트에서의 AFMoE. `models/afm-4.5b`가 로컬에 있지만 돌리지 않았다. snapshot reuse가 없는 세 번째 model-owned 계열이지만, 할당된 backend가 이미 `ModelOwned`이고 기부가 이미 건너뛰어지던 dense backend에 머무르므로, 게이트는 한 early return을 동등한 더 이른 것으로 바꿀 뿐이다. 구성상 동작 불변이나 측정되지 않았다.

CUDA가 필요한 OpenXLA CI 잡 2개. 이 diff와는 무관하다.

## 8. 학습 포인트

- 술어가 될 수 있는 값이 셋이고 그중 하나가 **거짓말하는 값**이라면, 수정은 체크를 추가하는 것이 아니라 셋 중 옳게 고르는 것이다. 같은 이유로 같은 선택을 이미 한 트리 안 선례를 찾아라. 여기엔 둘 있었다.
- **가드 수와 테스트 수가 맞는 것은 커버리지가 아니다.** 각 테스트가 서로 다른 가드에 대응하는지, 특히 각 가드를 지웠을 때 최소 하나가 실패하는지 확인한다. 테스트는 자기가 덮는 것처럼 보이는 가드와 무관한 이유로 초록일 수 있다.
- **반사실 검증은 그것이 훑는 가드-테스트 매핑만큼만 완전하다.** 이번 것은 가드 넷을 검증하고 다섯째를 조용히 건너뛰었다.
- 고치려는 실패만이 아니라 **수정의 특징적 실패**를 측정한다. 작업을 거절하는 게이트라면, 거절하지 말아야 할 곳에서 그 작업이 여전히 일어남을 증명해야 한다.
- 방어 심층 주장은 정확하게 적는다. "이 다섯 중 어느 하나만으로도"는 다섯 중 둘에 대해 거짓이었고, **부정확한 안전 주장은 소박한 주장보다 나쁘다.** 다음 독자가 다른 항이 덮어 준다고 믿고 한 항을 지울 수 있기 때문이다.
