# PR #2299: Bailing MoE Linear no-ID prefill 계약 수정

**작성일**: 2026-10-14

**상태**: 완료

**위험도**: 중간 (prefill 정책 및 긴 prompt의 transient memory)

## 요약

Bailing MoE Linear가 이제 `supports_chunked_prefill() == false`를 선언합니다. Sequence ID가 없는 다중 토큰 호출은 독립적인 prompt 사이에 history가 섞이지 않도록 model-owned recurrent 및 KV state를 의도적으로 교체하므로, no-ID 호출자에게 chunk continuation을 제공하면 앞선 chunk를 잃을 수 있습니다.

## 문제

`forward_for_sequence`가 모든 다중 토큰 no-ID forward 시작 시 fallback state를 초기화하는데도 이 모델은 trait의 chunked-prefill 지원 기본값을 상속했습니다. 현재 DirectEngine은 안정적인 sequence ID를 사용해 이 초기화를 피하므로, 이번 변경은 이전 DirectEngine 상태 처리가 prompt를 잘랐다고 주장하지 않습니다.

## 설계

- 독립 no-ID prompt 초기화와 stable-ID continuation을 보존하면서 capability를 override합니다.
- 기존의 0이 아닌 결정적 six-layer fixture를 `BailingMoeLinearModel::from_weights`로 불러옵니다. 이 fixture는 recurrent linear-attention layer 네 개와 global-attention layer 두 개를 포함합니다.
- no-ID 2-token 및 3-token forward를 모두 평가한 뒤 layer 0 offset이 5가 아닌 3인지 확인합니다.
- 준비된 stable ID로 2-token 및 3-token forward를 모두 평가한 뒤 같은 recurrent layer가 offset 5까지 이어지는지 확인합니다.

## 검증

Root 및 surgery CUDA `test-fast` build graph가 14분 20초에 통과했습니다. GPU lock 아래에서 실행한 `bailing_moe_linear_loaded_model_chunking_contract_is_safe` 회귀 테스트도 1개 중 1개가 0.78초에 통과했습니다. 소스는 `git diff --check`와 독립 구현 검토도 통과했습니다. 승인 fixture는 synthetic checkpoint shape이며 외부 checkpoint 실행을 주장하지 않습니다.

## 트레이드오프

Capability는 sequence ID 유무에 따른 지원 여부를 표현할 수 없습니다. DirectEngine의 stable-ID 상태 처리는 안전하지만 DirectEngine을 포함해 capability를 따르는 모든 호출자가 whole-prompt prefill plan을 사용합니다. 따라서 긴 prompt에서 chunk-bounded prefill보다 peak transient memory가 늘 수 있습니다.

## 변경 요약

모델에 capability override 하나를 추가하고 기존의 0이 아닌 결정적 fixture fill을 helper로 분리하며 새 의존성 없이 loaded-model state-contract 회귀 테스트 하나를 추가합니다.

## 최종 통합 검증

MLX `243bdf09`, mlxcelverse, 앞서 머지한 추론 수정을 포함한 main `b159c89d`를 소스 충돌 없이 통합했다. 다시 빌드한 CUDA `test-fast` binary에서 `models::bailing_moe_linear` 테스트 36개가 모두 통과했다. 가중치를 적재한 6-layer 요청 상태 회귀 테스트를 포함하며, 최종 pin에서 ID 없는 prompt의 초기화와 같은 ID의 연속 처리를 새로 검증했다. Formatting과 diff 검사도 통과했다. 합성 가중치를 적재한 모델을 사용했으며 외부 체크포인트 추론을 주장하지 않는다.
