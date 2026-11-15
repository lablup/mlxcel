# 기술 리포트: PR #2973 - 선택형 주기적 prefill checkpoint

**작성일**: 2026-11-12
**상태**: 완료 (#2885 Part B의 일부; 간격 기반 eviction은 남음)
**언어**: Rust, Python
**위험도**: Medium

## 요약

PR #2973은 snapshot 전용 모델에 세 번째 prompt-cache snapshot 종류를 추가한다. History boundary 앞에서 일정 간격(기본 16,384 토큰)의 배수마다 찍는 checkpoint다. 긴 컨텍스트 깊은 곳에서 history를 고치는 agent는 이제 system-prompt root가 아니라 수정 지점 직전의 checkpoint에서 이어서 계산한다. Turn 9가 28K 프롬프트의 약 20K 지점을 고치는 Qwen3.8-Flash-Next replay에서 그 턴은 4,744 대신 16,384 토큰을 재사용했고, TTFT는 22.16 s에서 11.25 s로 줄었다. 스위치 `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL`은 기본값이 꺼짐이다.

## 1. 문제 정의

Snapshot 전용 family(recurrent 또는 model-owned 상태)는 새 프롬프트가 저장된 토큰과 정확히 같은 토큰으로 시작할 때만 상태를 재사용한다. #2950 이후 저장소에는 prefill snapshot이 두 종류 있었다. 같은 채팅의 다음 턴을 위한 history boundary와, 같은 system 블록으로 시작하는 새 채팅을 위한 system-prompt root다. 28,000 토큰 채팅의 20,000번째 토큰 근처를 고치거나 compaction하면 history snapshot은 쓸 수 없고, 4,744 토큰의 root에서 시작하면 약 23K 토큰을 다시 prefill해야 한다.

이슈 본문은 checkpoint를 prefill 청크 경계에 맞춰 청크를 나누지 않도록 하자고 제안했는데, 여기에는 맞지 않는다. Snapshot family의 history 구간은 #1143 이후 청크로 나뉘지 않는 한 번의 forward라서 맞출 청크 경계가 없다. 그래서 checkpoint는 plan에 새 분할 지점으로 들어가야 한다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `server/prompt_cache/policy.rs` | `checkpoint_interval_tokens()`: 기본 꺼짐, `1`/`on`은 16,384, 256 이상 정수는 그 간격; boundary snapshot이 꺼져 있으면 무시 |
| `server/batch/scheduler/mod.rs` | `checkpoint_points(interval, start, upper, min_prefix, max)`: 채택되지 않은 구간 안의 절대 배수, 가장 깊은 `max`개 유지 |
| `server/batch/scheduler/prompt_cache.rs` | `snapshot_splits`가 root와 history boundary 자리를 먼저 잡고 checkpoint를 추가; `insert_boundary_snapshot`이 root, history, checkpoint로 나눠 저장 |
| `mlxcel-core/src/prefill_plan.rs` | `MAX_SPLITS` 4 → 16, public으로 변경 |
| `prompt_cache/entry.rs`, `store.rs` | `SnapshotOrigin::Checkpoint`, session-chain supersede 규칙에서 제외 |
| 테스트 | policy 파서, `checkpoint_points`, 저장소 supersede 제외, history를 고친 요청이 checkpoint를 채택하고 cold 실행을 재현하는 scheduler 테스트 |
| 문서, 스크립트 | `CONTINUOUS_BATCHING.md`, `environment-variables.md`, 벤치마크 결과, replay 스크립트의 `--edit-turn`/`--edit-target` |

## 3. 기술적 선택과 그 이유

### 3.1 채택 지점 기준 오프셋이 아니라 절대 배수

캐시 hit이 cold 실행을 재현하려면 채택한 prefix가 cold plan의 분할 지점에서 끝나야 한다. 0번 토큰부터 센 간격의 배수는 프롬프트 prefix에만 의존하므로, 처음 `k * interval` 토큰이 같은 이후 프롬프트도 cold로 돌 때 같은 지점에서 나뉜다. 채택 지점부터 세면 분할 지점이 캐시 상태에 따라 달라져서 이 성질이 깨진다.

### 3.2 Supersede 제외

저장소는 한 대화에서 같은 종류의 snapshot을 가장 긴 것 하나로 줄여서 채팅마다 boundary snapshot을 하나만 둔다. 이 규칙을 checkpoint에 적용하면 32K checkpoint가 16K checkpoint를 지우는데, 두 지점 사이를 고친 요청에 필요한 것이 바로 16K 쪽이다. 그래서 checkpoint 수는 바이트 예산의 LRU로만 제한한다.

### 3.3 Root와 history 자리 예약

기존 상한 4개는 얕은 지점을 남기는 방식이라 64K 프롬프트에서 history boundary가 빠질 수 있었다. 상한을 16으로 올렸고, scheduler가 root와 boundary 자리를 먼저 잡은 뒤 가장 깊은 checkpoint를 남기므로 multi-turn hit은 사라지지 않는다. 대신 간격 14개(기본값으로 229K 토큰)를 넘는 프롬프트는 가장 얕은 checkpoint를 버린다.

## 4. 검증

- 단위 테스트: 지점 계산, 파서, supersede 제외, tiny Gemma 3 fixture에서 청크 유무 두 경우의 `edited_history_adopts_the_last_checkpoint_and_reproduces_its_cold_run`.
- `-D warnings` clippy와 fmt 통과; CI 통과.
- Replay(`docs/benchmark_results/qwen4-exp-prefill-checkpoints-replay-m3ultra-2026-11-12.md`): TTFT 합 70.18 s → 58.66 s, 캐시된 토큰 268,730개 중 197,007 → 208,647.

Replay에서 일반적인 체인 턴(A5~A8)이 `qwen4_exp`에서 checkpoint 유무와 관계없이 첫 토큰 기준 약 0.4 nat만큼 cold 실행과 다를 수 있다는 것도 확인했다. 기존부터 있던 문제라 별도로 추적한다.

실행하지 않음: CUDA/ROCm, 메모리가 작은 Mac(#2883 가드레일), 전체 workspace 테스트.

## 5. 후속 조치

- #2885 Part B PR3: snapshot 사이 간격이 가장 작은 것부터 지우는 eviction.
- 기본 2 GB snapshot 예산에는 이 모델의 16K~32K checkpoint가 두 개 정도만 들어간다(하나에 0.5~1.2 GB). 기본값으로 이 스위치를 쓸모 있게 하려면 모델별 기본 예산이 필요할 수 있다.
- `qwen4_exp`의 체인 hit 편차는 별도 이슈로 등록.
