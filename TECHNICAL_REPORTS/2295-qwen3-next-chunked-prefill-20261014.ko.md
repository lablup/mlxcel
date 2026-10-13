# 기술 보고서: PR #2295 — Qwen3-Next chunked-prefill opt-out

**작성일**: 2026-10-14

**상태**: 완료

**위험도**: 중간

## 요약

Qwen3-Next가 이제 `supports_chunked_prefill() == false`를 명시한다. 이에 따라 cache-less `LanguageModel` 호출자가 prompt를 sequence ID 없는 여러 forward로 나눠 각 호출마다 model-owned GatedDelta 및 KV state를 초기화하는 경로를 선택하지 않는다. production weight constructor로 불러온 실제 1-layer GatedDelta 모델이 capability와 두 state 경로를 검증한다.

## 1. 문제 정의

`Qwen3NextModel::forward_for_sequence`는 sequence ID가 없는 multi-token forward를 매번 새 prompt로 해석한다. 이 초기화는 독립적인 cache-less prompt를 격리하지만, no-ID chunking 호출자는 앞 chunk가 만든 state를 다음 chunk에서 버리게 된다. 기존 trait 기본값은 이 계약과 달리 chunking 지원을 선언했다.

현재 DirectEngine prefill은 모든 piece에 동일한 sequence ID를 전달하므로 per-sequence state를 유지한다. 이번 변경은 현재 server state lifecycle을 바꾸지 않고 legacy 및 다른 cache-less 호출자를 위한 보수적인 trait capability를 정의한다.

## 3. 기술적 선택과 그 이유

`ModelOwnedSequenceState` 계약을 바꾸지 않고 모델이 opt-out하도록 선택했다. 명시적인 no-ID continuation mechanism은 독립 prompt 사이에서 계속 초기화하면서도 continuation을 구분하는 여러 family 공통 lifecycle signal이 필요하다. 국소적인 opt-out은 이 격리를 보존하지만, 긴 prompt에서 chunk별 transient memory bound 대신 전체 prompt를 하나의 graph로 처리할 수 있다.

## 7. 변경 요약

- no-ID state 동작의 근거와 DirectEngine 경로의 차이를 설명하는 Qwen3-Next `supports_chunked_prefill` override를 추가했다.
- `Qwen3NextModel::from_weights`로 실제 1-layer GatedDelta fixture를 불러올 수 있도록 deterministic nonzero weight를 추가했다.
- no-ID prompt forward 두 번을 강제 평가해 두 번째 prompt의 cache offset이 3으로 초기화되는지 검증하고, 동일한 `SequenceId`의 forward 두 번은 offset 5까지 이어지는지 검증했다.

## 검증

CUDA `mlxcel` 및 `mlxcel-surgery` library test graph가 성공적으로 빌드됐다. GPU lock 아래 focused `qwen3_next_loaded_model_chunking_contract_is_safe` test가 0.99초에 1/1 통과했다. Rust formatting과 diff check가 통과했으며 독립 implementation review에서 correctness 문제가 없었다.

필수 repository contract가 모두 통과했으며 Python test 123개와 WebUI contract test 49개를 포함한다. 최초 timing-sensitive GPU guard sample은 native compile이 동시에 진행되는 동안 짧은 foreign process를 놓쳤지만, 코드 변경 없이 isolated guard retry와 broad contract rerun이 통과했다.

## 8. 후속 조치

- 부모 통합 단계에서 승인된 모든 후속 unit이 반영된 뒤 결합된 issue set을 검증한다.

## 참고 자료

- Issue #2285
- Issue #1749 및 PR #2293
- Issue #672 및 PR #1741
