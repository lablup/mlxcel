# PR #2300: RecurrentGemma no-ID prefill 계약 수정

**작성일**: 2026-10-14

**상태**: 완료

**위험도**: 중간 (prefill 정책 및 긴 prompt의 transient memory)

## 요약

RecurrentGemma가 이제 `supports_chunked_prefill() == false`를 선언합니다. Sequence ID가 없는 다중 토큰 호출은 독립적인 prompt 사이에 history가 섞이지 않도록 model-owned RGLRU state를 의도적으로 교체하므로, no-ID 호출자에게 chunk continuation을 제공하면 앞선 chunk를 잃을 수 있습니다.

## 문제

`forward_for_sequence`가 모든 다중 토큰 no-ID forward 시작 시 fallback state를 초기화하는데도 이 모델은 trait의 chunked-prefill 지원 기본값을 상속했습니다. 현재 DirectEngine은 안정적인 sequence ID를 사용해 이 초기화를 피하므로, 이번 변경은 이전 DirectEngine 상태 처리가 prompt를 잘랐다고 주장하지 않습니다.

## 설계

- 독립 no-ID prompt 초기화와 stable-ID continuation을 보존하면서 capability를 override합니다.
- 0이 아닌 결정적 one-layer recurrent Griffin 모델을 `GriffinModel::from_weights`로 불러옵니다.
- Single-token forward로 fallback convolution 및 RGLRU state를 채운 뒤 새 multi-token no-ID prompt가 fresh model의 동일 prompt와 일치하는지 검증합니다.
- 하나의 stable sequence ID로 5-token prompt를 2개와 3개로 나누고 continuation logits를 one-pass prompt의 동일 suffix와 비교합니다. 절대 또는 상대 허용 오차는 `1e-3` 미만입니다.

## 검증

Root 및 surgery CUDA `test-fast` build graph가 14분 25초에 통과했습니다. GPU lock 아래에서 실행한 `recurrent_gemma_` focused suite도 loaded-model contract 및 기존 model-owned state 테스트를 포함해 5개 중 5개가 0.84초에 통과했습니다. 소스는 `git diff --check`도 통과했습니다. 승인 fixture는 synthetic checkpoint shape이며 외부 checkpoint 실행을 주장하지 않습니다.

## 트레이드오프

Capability는 sequence ID 유무에 따른 지원 여부를 표현할 수 없습니다. DirectEngine의 stable-ID 상태 처리는 안전하지만 DirectEngine을 포함해 capability를 따르는 모든 호출자가 whole-prompt prefill plan을 사용합니다. 따라서 긴 prompt에서 chunk-bounded prefill보다 peak transient memory가 늘 수 있습니다.

## 변경 요약

Rust 모델 파일 하나에 capability override, 결정적 loaded-model fixture, state-contract 회귀 테스트 하나를 새 의존성 없이 추가합니다.

## 최종 통합 검증

MLX `243bdf09`, mlxcelverse, 앞서 머지한 모델 수정을 포함한 main `5f334629`를 소스 충돌 없이 통합했다. 다시 빌드한 CUDA `test-fast` binary에서 `models::recurrent_gemma` 테스트 7개가 모두 통과했다. 비영 가중치를 적재한 RGLRU fixture를 포함하며, 최종 pin에서 convolution/recurrent 상태 생성, ID 없는 prompt 초기화, 같은 ID의 분할 logits와 한 번에 실행한 결과의 일치를 새로 검증했다. Formatting과 diff 검사도 통과했다. 외부 체크포인트 추론 결과를 주장하지 않는다.
