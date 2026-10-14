# PR #2297: RWKV7 no-ID chunked prefill 계약 수정

**날짜**: 2026-10-14
**상태**: 구현 완료, CUDA build 및 focused GPU 검증 통과
**위험도**: 중간 (prefill 정책과 recurrent 출력 shape 수정)

## 요약

이 변경은 RWKV7이 cache-level chunked prefill을 지원하지 않는다고 선언합니다. Sequence ID가 없는 다중 토큰 호출은 서로 독립적인 프롬프트를 보장하기 위해 fallback recurrent state를 의도적으로 교체하므로, legacy no-ID 호출자에게 chunk continuation을 제공하면 앞선 chunk를 잃을 수 있습니다. 현재 DirectEngine 호출은 안정적인 sequence ID를 전달하며 기존 per-sequence model-owned state를 계속 사용합니다.

Loaded-model 회귀 테스트는 기존 RWKV7 recurrent 출력의 shape 결함도 드러냈습니다. 이제 각 step의 matrix product가 만든 singleton column 축을 제거한 뒤 timestep을 쌓습니다.

## 문제

RWKV7은 no-ID 경로가 모든 다중 토큰 forward 시작 시 recurrent state를 초기화하는데도 기본값인 `supports_chunked_prefill() == true`를 상속했습니다. 하나의 프롬프트를 여러 no-ID chunk로 나누는 호출자는 앞선 chunk 없이 계산된 logits를 받을 수 있었습니다.

Focused GPU 검증 중 실제 one-layer fixture가 `(2,4)`와 `(1,1,2,4,1)` shape 사이의 native broadcast 예외로 중단됐습니다. 이 문제는 fixture geometry가 아니라 production forward의 결함이었습니다. `wkv7_step_ops`는 `state @ r[..., None]` 결과를 `[B, H, D, 1]`로 만들었고, 각 step 출력을 그대로 쌓아 `[B, T, H, D, 1]`을 만들었습니다. `LayerNormPerHead`는 `[H, D]` weight가 batch와 time 축에 broadcast될 수 있도록 `[B, T, H, D]` activation을 기대합니다.

## 설계

- 기존 독립 프롬프트 초기화 동작을 유지하면서 `supports_chunked_prefill()`이 `false`를 반환하도록 override합니다.
- 각 recurrent step 출력의 마지막 singleton 축만 제거합니다. 출력 계약은 step마다 `[B, H, D]`, stacking 뒤 `[B, T, H, D]`가 되고 recurrent state는 `[B, H, D, D]`를 유지합니다.
- 0이 아닌 결정적 synthetic weight로 실제 one-layer 모델을 구성합니다.
- 단일 토큰으로 fallback state를 오염시킨 뒤 새 다중 토큰 no-ID 프롬프트가 새 모델의 동일 프롬프트와 같은지 검증합니다.
- 하나의 안정적인 sequence ID로 프롬프트를 두 번에 나누어 실행하고 continuation logits를 one-pass 프롬프트의 동일 suffix와 비교합니다.

## 검증

최초 root 및 surgery CUDA `test-fast` build graph는 14분 26초에 통과했습니다. 이어진 첫 focused GPU 실행은 실제 RWKV7 forward 경로에 도달해 위 native broadcast 예외로 shape 결함을 재현했습니다. 수정 후 CUDA graph는 2분 22초에 다시 빌드됐습니다. GPU lock 아래 `rwkv7_` suite는 5개 테스트 모두 통과했고, 실제 one-layer loaded-model contract 회귀는 0.64초에 통과했습니다. Fixture는 production loader를 통해 0이 아닌 결정적 synthetic weight를 적재하며, 외부 체크포인트 실행을 수행하거나 주장하지 않습니다. `git diff --check`는 통과했습니다.

전체 contract gate도 통제된 재실행을 거쳐 통과했습니다. 최초 실행은 Python tooling 이전의 모든 단계를 통과했지만, concurrent native build와 외부 부하가 호스트를 포화시킨 동안 timing-sensitive ROCm guard fixture에서 2.5초 monitor sample 누락 1건과 holder metadata `FileNotFoundError` 1건이 발생했습니다. Product 및 test source는 변경하지 않았습니다. 경쟁 build의 우선순위를 낮춘 뒤 해당 guard 테스트 두 개가 모두 통과했고, 전체 Python tooling 123개와 WebUI 49개 테스트도 통과했습니다.

## 의존성 갱신

위 검증 결과는 브랜치가 PR #2298의 main commit `4b0c4519`를 병합하기 전에 기록했습니다. Merge commit `9b05507b`는 새 MLX pin과 CUDA/Metal overlay base를 가져오지만 `src/models/rwkv7.rs`를 바꾸지 않았고, 해당 파일의 SHA-256은 `fa18e8e5e33f4af42fe7c7c8574e55db8578f90724d3a70d0cad4a7cf10adea7`로 유지됩니다. 최종 pin 통합 후 새 focused runtime 검증 결과는 아래에 기록합니다.

## 트레이드오프

이제 DirectEngine과 cache-less caller를 포함해 model capability를 따르는 모든 호출자가 whole-prompt prefill plan을 사용합니다. 긴 프롬프트에서는 chunk-bounded prefill보다 peak transient memory가 늘 수 있습니다. DirectEngine은 계속 안정적인 sequence ID를 제공하며, 이 변경은 기존 DirectEngine 상태 처리가 프롬프트를 잘랐다고 주장하지 않습니다.

## 변경 요약

Rust 모델 파일 하나를 변경합니다. Production 변경은 capability override와 recurrent 출력 축 수정 하나이며, 새 의존성 없이 실제 loaded-model contract test 하나를 추가합니다.

## 최종 통합 검증

MLX `243bdf09`, mlxcelverse 경로 변경, Kimi 추론 수정을 포함한 main `d3cf28ac`을 소스 충돌 없이 통합했다. 다시 빌드한 CUDA `test-fast` unit binary에서 `models::rwkv7` 테스트 5개가 모두 통과했다. 비영 가중치를 적재한 계약 회귀 테스트를 포함하며, 최종 pin에서 출력 형태, ID 없는 prompt의 상태 초기화, 같은 ID를 사용하는 분할 연속 처리를 재검증했다. 위에서 대기로 기록한 pin 갱신 후 검증은 이 실행으로 완료됐다. Formatting과 diff 검사도 통과했다. 외부 체크포인트 추론 결과를 주장하지 않는다.

갱신된 Clippy 검사에서 테스트 fixture의 짝수 판정 표기 경고도 발견했다. `% 2 == 0`을 `is_multiple_of(2)`로 바꿨으며 생성되는 가중치는 동일하다.
