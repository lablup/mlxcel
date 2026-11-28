# PR #2139: Completion snapshot을 상태가 실제로 가진 토큰 기준으로 저장

**작성일**: 2026-10-06
**상태**: CUDA(GB10)에서 Gemma 4, Gemma 3, GatedDeltaNet hybrid, dense KV family로 검증. Speculative burst donation은 측정하지 않음
**위험도**: 중간 (모든 family의 donated 엔트리 길이가 바뀜. model-owned family의 decode lookahead를 끔)

## 요약

Decode는 샘플링한 토큰을 다음 step에서 forward한다. 그래서 merged-EOS 종료를 제외한 모든 종료에서 마지막 생성 토큰은 모델 상태에 들어가지 않는다. 그런데 completion donation은 엔트리를 `prompt ++ generated` 기준으로 저장했고, exact-entry restore는 설치한 상태보다 한 위치 뒤에서 prefill을 재개했다. 이 PR은 실제로 소비된 토큰만 donate한다(`SequenceInfo::generated_in_state`). 또한 teardown trim이 상태에 닿지 않는 model-owned family에서는 decode lookahead pipeline이 돌지 않게 막는다. 이 trim 누락 때문에 Gemma 3 snapshot은 반대 방향으로, 주장하는 길이보다 한두 위치를 더 갖고 있었다.

Closes #1754.

## 1. 문제 정의

이슈 #1754는 코드 분석으로 off-by-one을 예측했다. 이후 댓글은 llama-4-scout chat turn이 cold 실행과 바이트 단위로 같았다고 보고해 이 예측과 어긋나 보였다. Capture 시점에 snapshot의 layer별 offset을 읽어 결론을 냈다.

| 종료 | 수정 전 | 수정 후 |
|---|---|---|
| `length`, stop 문자열 (Gemma 4, sync decode Gemma 3, Llama 3.2 KV) | offset = tokens - 1 | 일치 |
| merged EOS (Gemma 4) | 일치 | 일치 |
| `length`, 기본 플래그 (Gemma 3, lookahead 활성) | offset = tokens + 1 | 일치 |

Chat route에서는 이 결함이 가려진다. Turn 2가 assistant 응답을 다시 렌더링하므로(템플릿이 end-of-turn scaffold를 붙이고, 재토큰화가 항상 canonical하지는 않음) `prompt ++ generated`가 보통 turn 2의 prefix가 되지 않는다. llama-4-scout 댓글의 restore도 1817토큰 completion 엔트리가 아니라 1822토큰 warm-up 엔트리를 썼다. 반면 raw completion route는 completion 엔트리에 정확히 도달한다. 가장 긴 snapshot을 복원한 뒤 delta만 forward하는 warm-up extend도 마찬가지다.

## 2. 변경 내용

- `decode_tick.rs`의 merged-EOS 지점 세 곳에서 `SequenceInfo::eos_terminated`를 설정한다. 이 경우 EOS는 샘플링되지만 push되지 않는다. `SequenceInfo::generated_in_state()`는 이 경우 생성 토큰 전체를, 그 밖에는 마지막 토큰을 뺀 나머지를 반환한다.
- Classic decode finalize와 prefill 중 종료 donation이 `generated_in_state()`를 넘긴다. eos-at-prefill donation은 원래 빈 slice를 넘긴다.
- `lookahead_params`는 `model.sequence_state_layout().backend == ModelOwned`이면 `None`을 반환한다. 이전 검사는 allocated backend를 읽었는데, paged override는 Gemma 3, AFMoE, Llama 4의 cache가 실제로는 `ModelOwnedSequenceState`에 있는데도 이를 `PagedKvCache`로 할당한다. `apply_lookahead_trim`이 순회하는 `get_caches_mut`는 이 family들에서 비어 있으므로 speculative append가 되돌려지지 않았다.
- Speculative burst donation은 committed tail 전체를 그대로 넘긴다. Burst의 마지막 토큰이 상태에 있는지는 drafter verify rollback에 달려 있는데, 이를 측정하지 않았다.

## 3. 기술적 선택: restore가 아니라 donation을 수정

이슈는 두 가지 수정안을 제시했다. 하나는 토큰 하나를 덜 donate하는 것, 다른 하나는 복원된 offset에서 prefill을 재개하는 것이다. Recurrent family(GatedDeltaNet, Mamba)에는 재개 기준으로 쓸 offset이 없으므로, 모든 family에 똑같이 적용할 수 있는 쪽은 donation 수정뿐이다. 판단 기준은 `FinishReason`이 아니라 종료를 일으킨 샘플 토큰이 push되었는가이다. Structured-output stop도 `Stop`으로 끝나지만 토큰을 push한다.

## 4. 검증 (GB10, CUDA, `mlxcel-server`)

프로토콜: turn 1은 `/v1/completions`로 실행하고, turn 2는 turn 1 + 응답 + suffix로 만든다. Warm turn 2는 같은 프로세스에서 completion 엔트리를 정확히 복원해 실행하고, cold turn 2는 새 프로세스에서 실행한다. 양쪽 모두 `--prefill-chunk-size 1`을 써서 forward 폭에 따른 반올림 차이를 없앴다. 수정 전 arm은 같은 바이너리에서 임시 스위치로 이전 동작을 켠 것이며, 스위치는 커밋 전에 제거했다.

| 체크포인트 / 종료 | 수정 전 | 수정 후 |
|---|---|---|
| gemma-4-12b `length` | 2번째 토큰에서 분기, 첫 logprob -0.125 vs -0.625 | 동일, max abs dlogprob 0.0 |
| gemma-4-12b stop 문자열 | 텍스트 동일, abs dlogprob 최대 0.625 | 동일, 0.0 |
| gemma-4-12b EOS | 동일 | 동일 |
| gemma-3-4b `length`, stop 문자열 | 1번째 토큰에서 분기 | 동일, 0.0 |
| qwen3.5-0.8b `length` | 텍스트 동일, abs dlogprob 0.125 | 동일, 0.0 |
| llama-3.2-1b dense KV `length` (`--parallel 1 --apc-block-size 1`) | 1번째 토큰에서 분기, 첫 logprob -2.97 vs -2.07 | 동일, 0.0 |

Gemma 3 4B lookahead (arm별 4회, 200토큰, 기본 플래그):

- 수정 전: 출력이 `MLXCEL_FORCE_SYNC=1`과 달랐고 decode 속도는 84.1-84.5 tok/s였다.
- 수정 후: force-sync와 바이트 단위로 같고 decode 속도는 74.1-75.2 tok/s다.
- Llama 3.2(KV)에서는 lookahead 출력이 force-sync와 같다. 따라서 Gemma 3의 차이는 pipeline 자체가 아니라 되돌려지지 않은 append 때문이다.
- Sequence-aware trim hook(#1755)이 들어오면 이 family들에서 pipeline을 다시 켤 수 있다.

단위 테스트(`src/server/batch/scheduler_completion_snapshot_tests.rs`, tiny 실제 Gemma 3):

- decode step을 거친 `length` 종료
- 생성 토큰 1개 경계(prefill 중 종료)
- EOS 종료
- decode 중인 model-owned sequence가 lookahead params를 받지 않는지 확인

각 테스트는 저장된 엔트리의 토큰 수를 그 snapshot이 가진 offset과 비교한다. 이전 동작에서는 네 개 중 세 개가 실패한다.

## 5. 이 호스트에서 검증하지 못한 항목

- Speculative burst donation: drafter 실행을 측정하지 않았다.
- Cancelled 종료: 같은 규칙(마지막으로 push된 토큰은 forward되지 않음)을 따르지만 HTTP로 실행해 보지는 않았다.
- llama-4-scout: 이 호스트의 모델당 40 GB 한도를 넘는다.

## 6. 관련 항목

- #1760 / PR #2136: 프롬프트 전체 히트 수정. 이 PR의 측정에서 `length` off-by-one이 처음 드러났다.
- #1755: model-owned prefill trim을 위한 sequence-aware trim hook.
- #1346: 이 gate가 적용한 natural backend 대 allocated backend 교훈.
