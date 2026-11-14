# PR #3037: 직전 턴의 복원 지점 유지

**작성일**: 2026-11-12
**상태**: 완료 (#3035 종료, #2885 참조)
**위험도**: 낮음 (opt-in 스위치, 기본 동작 변경 없음)

## 요약

snapshot 전용 모델 계열에서 agent가 마지막 user message를 바꾸거나 compaction하면, prompt cache는 system-prompt root 이상을 재사용하지 못했습니다. 이 수정에 필요한 snapshot은 직전 턴이 복원에 쓴 snapshot인데, 직전 턴이 자기 snapshot을 저장하는 순간 session-chain supersede 규칙이 그것을 지웠기 때문입니다. `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN=1`은 세션마다 이 snapshot 하나를 남깁니다. Qwen3.8-Flash-Next에서 모델 기반 기본 예산으로 측정하면, 수정된 턴은 4,744 토큰 재사용에 TTFT 22.47 s였던 것이 25,804 토큰 재사용에 2.90 s가 되어 수정 없는 턴과 같아집니다.

## 1. 문제 정의

### 1.1 배경

snapshot 전용 계열(Qwen3.8 Flash Next, Mamba hybrid)은 recurrent state를 잘라 쓸 수 없어서, prompt cache가 prefix 전체 단위로 snapshot을 저장합니다. prefill 중에 잡는 history boundary, 생성이 끝날 때 넣는 completion, opt-in인 system-prompt root와 주기적 checkpoint가 있습니다. #1146 이후 store는 같은 세션에서 더 긴 같은 종류의 snapshot이 들어오면 그 prefix인 짧은 snapshot을 supersede하고, 그래서 대화 하나는 boundary 하나와 completion 하나만 유지합니다.

### 1.2 기존 문제점

- 턴 N은 턴 N-1의 snapshot에서 복원하고, 턴 N의 snapshot을 저장하면 그 snapshot이 supersede됩니다. 턴 N의 user message를 고친 요청에는 바로 그 지워진 상태가 필요합니다. 턴 N의 boundary는 수정된 message 뒤에서 끝나므로 수정된 요청의 prefix가 아닙니다.
- 모델 기반 기본 예산은 8,192 토큰짜리 snapshot 6개 기준(이 모델에서 2.1 GB)인데, 28K 토큰에서는 snapshot 하나가 약 1 GB라 checkpoint도 쓰이기 전에 LRU로 빠집니다. checkpoint는 8 GiB 예산에서만 이 수정을 받아 16,384 토큰을 재사용했습니다(11.6 s).
- 이전 replay는 assistant 응답을 미리 정한 문장으로 넣어서 completion snapshot이 다음 턴과 일치하지 않았고, completion이 쌓이는 것처럼 보여 snapshot 부담이 과장됐습니다.

## 2. 기술적 선택과 그 이유

### 2.1 고정 깊이 대신 복원 출처를 유지

| 방안 | 장점 | 단점 |
|---|---|---|
| 같은 종류 조상 중 가장 긴 둘을 유지(깊이 2) | supersede 필터 안의 단순한 규칙 | boundary와 completion 두 체인이 모두 늘어 대화당 4개가 되고, completion이 이미 포함하는 boundary까지 남김 |
| Strata식 메시지 경계 추가 캡처 | lookup과 무관 | 턴마다 prefill 분할과 state 복사가 하나 더 필요 |
| **선택: 세션별로 lookup이 채택한 snapshot을 기록하고 supersede에서 제외** | 직전 턴이 시작한 상태를 종류와 무관하게 정확히 유지(실제 응답이면 completion, 재토큰화가 다르면 boundary), 추가 비용은 snapshot 하나 | store에 세션별 map이 하나 늘어남 |

제외 대상은 복원할 때마다 옮겨갑니다. 턴 N+1이 턴 N의 snapshot을 채택하면 턴 N-1의 snapshot은 보호를 잃고 다음 insert에서 supersede되므로, 정상 상태에서 대화당 snapshot은 세 개입니다.

### 2.2 opt-in과 그에 맞춘 기본 예산

스위치를 켜면 대화당 snapshot이 하나 늘어나고(이 모델 28K 토큰에서 약 1 GB), 그래서 기본값은 끔입니다. 스위치가 켜져 있고 예산이 모델 기반 기본값이면 목표 entry 수가 6에서 9로 올라가며(대화당 두 개가 세 개가 되므로), 여유 메모리의 4분의 1 상한은 그대로 적용됩니다. 운영자가 정한 상한은 건드리지 않습니다.

## 3. 구현 상세

- `PromptCacheConfig::keep_previous_turn`: startup이 sizing 전에 `keep_previous_turn_enabled()` 값으로 설정합니다.
- `Inner::restore_sources: HashMap<(sessionless bucket, session key), digest>`: session key가 있는 snapshot hit에서 기록하고, supersede 필터가 읽으며, 해당 entry가 `remove_snapshot`으로 빠지거나 `clear()`가 호출되면 지웁니다.
- `bounded_capacity`가 목표 entry 수를 인자로 받고, `recommend_model_snapshot_capacity`는 `model_aware_snapshot_target_entries`에서 그 값을 받습니다.
- replay 하네스의 `--live-replies`는 서버가 실제로 생성한 응답을 다음 턴 history에 붙입니다.

## 4. 결과

| 수정 위치 | 스위치 끔 | 스위치 켬 |
|---|---|---|
| 마지막 user message, 28,553 토큰 prompt | 4,744 재사용, 22.47 s | 25,804 재사용, 2.90 s |
| 3턴 전, 28,195 토큰 prompt | 4,744 재사용, 22.18 s | 4,744 재사용, 22.06 s |

스위치를 켜면 store는 7개 entry(3.15 GB 중 3.08 GB), 끄면 4개(2.10 GB 중 1.75 GB)로 끝납니다. 수정 없는 요청의 출력은 두 arm에서 같습니다. completion snapshot을 채택한 요청은 두 arm 모두 cold 실행과 출력이 다른데, completion의 뒷부분은 decode 단계가 쓴 상태라서 생기는 것으로 이 PR 이전부터 있던 동작입니다.

## 5. 변경 요약

| 영역 | 변경 |
|---|---|
| Store | 세션별 복원 출처 기록과 supersede 제외 |
| Sizing | 목표 entry 수 인자, 스위치를 켜면 9개 |
| Startup | sizing 전에 스위치를 config로 복사 |
| 테스트 | store 테스트 3개, sizing 테스트 1개 |
| 문서 | 환경 변수 항목, CONTINUOUS_BATCHING 문단, 원시 데이터를 포함한 벤치마크 결과 |
| 하네스 | `--live-replies` |

## 6. 후속 조치

- 한 턴보다 더 앞의 수정은 여전히 checkpoint와 큰 예산으로만 받을 수 있습니다. 긴 agent 대화에서 실제로 걸리는 제약은 기본 예산의 8,192 토큰 기준 길이입니다.
- completion snapshot 복원은 temperature 0에서도 cold prefill과 바이트 단위로 같지 않습니다. 고칠지 문서화할지는 정해지지 않았습니다.
- 작은 Mac에서의 메모리 동작(#2883)은 측정하지 않았습니다.
