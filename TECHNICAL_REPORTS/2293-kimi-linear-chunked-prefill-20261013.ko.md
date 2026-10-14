# 기술 보고서: PR #2293 — Kimi Linear chunked-prefill opt-out

**작성일**: 2026-10-13

**상태**: 런타임 후속 조치가 있는 검증 완료

**위험도**: 중간

## 요약

Kimi Linear가 이제 `supports_chunked_prefill() == false`를 명시한다. 이에 따라 cache-less `LanguageModel` 호출자가 prompt를 sequence ID 없는 여러 forward로 나눠 각 호출마다 model-owned Delta 및 MLA state를 초기화하는 경로를 선택하지 않는다. production weight constructor로 불러온 작은 모델이 trait interface를 통해 이 capability를 검증한다.

## 1. 문제 정의

`KimiLinearModel::forward_for_sequence`는 sequence ID가 없는 multi-token forward를 새 prompt로 해석한다. 이 초기화는 독립적인 두 cache-less 호출 사이의 state를 격리하지만, no-ID chunking 호출자는 앞 chunk가 만든 recurrent state를 다음 chunk마다 버리게 된다. 기존 shared trait 기본값은 이 계약과 달리 chunking 지원을 선언했다.

현재 DirectEngine prefill은 모든 piece에 동일한 sequence ID를 전달하므로 no-ID 초기화에 들어가지 않는다. 이번 변경은 보수적인 trait capability를 정의하고 legacy 또는 다른 cache-less 호출자가 안전하지 않은 분할을 선택하지 못하게 한다.

## 3. 기술적 선택과 그 이유

`ModelOwnedSequenceState` 계약을 바꾸지 않고 해당 모델이 opt-out하도록 선택했다. 명시적인 continuation mechanism은 여러 모델 family에 걸친 state 계약 재설계가 필요하며 independent-prompt reset도 보존해야 한다. 이 opt-out은 국소적이고 되돌리기 쉽지만, 긴 prompt에서 chunk별 transient memory bound 대신 전체 prompt를 하나의 graph로 처리할 수 있다.

## 7. 변경 요약

- no-ID state 동작의 근거와 함께 Kimi Linear `supports_chunked_prefill` override를 추가했다.
- zero-layer 및 tied-embedding fixture를 `KimiLinearModel::from_weights`로 불러와 `&dyn LanguageModel`을 통해 정책을 검증한다.
- 같은 reset/default-capability 조합을 가진 다섯 sibling을 다시 확인하고 #2285부터 #2289까지 등록했다.

## 검증

CUDA `mlxcel` 및 `mlxcel-surgery` library test graph build가 성공했다. GPU lock 아래 focused loaded-model test 1/1이 통과했다. formatting과 diff check가 통과했으며 독립 implementation review와 security/performance review에서 지적 사항이 없었다.

실제 checkpoint 검증에는 `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` revision `919175d75427f1c10cfc17e7254b53295edddf72`를 사용했다. baseline main `190b83c3`와 main `448935b6` 및 동결된 policy diff로 만든 candidate는 모두 3,411-token prompt의 64개 출력에서 기본 설정과 `MLXCEL_PREFILL_CHUNK=0` 양쪽 모두 token ID 0을 냈고, 21-token chunk-disabled 산술 prompt에서도 ID 0을 냈다. 동일하게 잘못된 출력은 accuracy 또는 numerical parity 통과가 아니다. candidate CLI SHA256은 `c0b0319f0b4f0302e9e5dccf02f4f403453ec3dc9aa7e5e1a4ce625f3f060eb7`이며 원인 진단과 exact-answer 승인 조건은 #2292에서 추적한다.

## 8. 후속 조치

- #2292에서 Kimi Linear CUDA 정상 inference를 복구하고 short 및 long prompt의 정확한 답을 증명해야 한다.
- #2285부터 #2289까지는 이 변경의 범위를 넓히지 않고 감사된 sibling no-ID 계약을 다룬다.

## 참고 자료

- Issue #1749
- Issues #2285, #2286, #2287, #2288, #2289, #2292
- Issue #672 및 PR #1741
