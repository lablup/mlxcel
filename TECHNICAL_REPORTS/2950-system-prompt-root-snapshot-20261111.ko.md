# 기술 리포트: PR #2950 - 선택형 system-prompt root snapshot

**작성일**: 2026-11-11
**상태**: 완료 (#2885 Part B의 일부; 주기적 checkpoint와 간격 기반 eviction은 남음)
**언어**: Rust, Python
**위험도**: Medium

## 요약

PR #2950은 prompt-cache snapshot의 두 번째 종류를 추가한다. system prompt와 tool 목록이 끝나는 지점(root)의 모델 상태다. Part A(PR #2911)는 history boundary에만 snapshot을 저장했기 때문에, 같은 system prompt로 시작하는 새 채팅이나 앞쪽 history를 고친 턴은 프롬프트 전체를 다시 prefill했다. `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1`이면 서버가 첫 prefill을 root에서도 나누고 두 snapshot을 모두 저장한다. Qwen3.8-Flash-Next(M3 Ultra)에서 요청 16개짜리 agent replay의 TTFT 합은 51.16 s에서 38.37 s로, 새 채팅 첫 턴은 약 5.6 s에서 약 1.5 s로 줄었다. 두 조건 모두 16개 응답이 cold 실행과 같았다. 기본값은 꺼짐이다. 38개 파일, +1923/-111.

## 1. 문제 정의

Agent 클라이언트는 요청마다 큰 고정 prefix(system prompt와 tool schema, 보통 4K~20K 토큰)를 보낸다. Part A는 직전 턴의 history boundary까지만 재사용한다. 흔한 두 경우가 완전히 빗나갔다.

- **같은 prefix의 새 채팅.** 이전 채팅의 history snapshot은 system prompt 직후에서 갈라지고, 그 지점에는 snapshot이 없었다.
- **History 편집이나 compaction.** Agent가 몇 턴 전 메시지를 고치면 history snapshot이 맞지 않아 프롬프트 전체를 다시 prefill한다.

임의의 prefix를 재사용하는 것은 안전하지 않다. Snapshot은 cold 실행의 prefill plan에도 있는 split 지점에서 찍혔을 때만 cold 실행을 재현한다(`docs/CONTINUOUS_BATCHING.md`). 그래서 root는 나중에 자른 prefix가 아니라 plan 안의 실제 split이어야 한다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `mlxcel-core/src/prefill_plan.rs` | 단일 `Option<usize>` boundary를 `PrefillSplits`(정렬된 split 최대 4개)로 교체; `plan_geometry`는 채택되지 않은 구간의 split만 남김 |
| `server/prompt_cache/policy.rs` | `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT` 파싱: 기본 꺼짐, `1`/`on`은 최소 프롬프트 2048 토큰, 정수는 최소 길이 |
| `server/chat_request.rs` | Root render: 앞쪽 system/developer 메시지와 sentinel user 턴을 렌더링한 뒤 sentinel에서 자름; history render의 strict prefix여야 함 |
| `server/batch/scheduler*` | `system_root_len`이 root split 계산; `snapshot_splits`가 두 split을 plan에 전달; `insert_boundary_snapshot`이 origin별로 저장 |
| `prompt_cache/entry.rs`, `store.rs` | `SnapshotOrigin::SystemRoot`; root가 snapshot 예산의 절반 이하인 동안 LRU가 root를 건너뜀 |
| Gemma 4 MTP 경로 | `PrefillSplits`를 써서 drafter가 serving과 같은 지점에서 나눔(#2160) |
| 문서, 스크립트 | `CONTINUOUS_BATCHING.md`, `environment-variables.md`, 벤치마크 결과, `scripts/bench_prompt_cache_replay.py` |

## 3. 기술적 선택과 그 이유

### 3.1 Sentinel user 턴으로 렌더링

System 메시지만 렌더링하면 user 턴을 요구하는 template에서 실패한다(Qwen3.8 template은 "No messages provided"를 던진다). 그래서 sentinel을 담은 user 메시지를 붙여 렌더링하고 sentinel에서 자른다. 결과가 전체 render보다 짧은 strict prefix일 때만 받아들이므로, system 블록을 재배치하거나 고치는 template에서는 root 없이 동작한다.

### 3.2 경계 토큰 하나를 버림

Root split은 root와 프롬프트의 최장 공통 prefix보다 한 토큰 앞이다. Root render의 마지막 토큰은 뒤 텍스트와 합쳐져 다르게 토큰화될 수 있어서, `system_root_len`은 한 토큰 물러나고 split이 history boundary보다 반드시 앞에 있도록 요구한다.

### 3.3 세션 범위, 공유하지 않음

Root는 요청의 cache 세션 안에만 머문다. 세션 간 공유는 multi-tenant 서버에 가장 이득이지만, 문서가 snapshot을 세션 간에 공유하지 않는다고 약속하고 있고 이를 바꾸는 것은 별도의 결정이다.

### 3.4 기본값 꺼짐

추가 split 때문에 첫 요청에 prefill 조각이 하나 늘고(A1: 6.13 s 대 6.06 s), 이 모델에서 root마다 snapshot 메모리가 약 250 MB 든다. 메모리가 작은 Mac은 측정하지 않았으므로 선택형으로 둔다.

## 4. 검증

- 대상 lib 테스트 98개: plan 테스트, root render, `system_root_len`, 새 채팅이 root를 채택하고 cold 실행을 재현하는지 보는 scheduler 테스트 2개 포함.
- `-D warnings` clippy와 fmt 통과; CI 통과.
- Replay(`docs/benchmark_results/qwen4-exp-system-root-replay-m3ultra-2026-11-11.md`): 캐시된 토큰 175,486개 중 122,978 → 137,210; snapshot hit 12/16 → 15/16; 두 조건 모두 16/16이 cold와 동일.

실행하지 않음: CUDA/ROCm, 메모리가 작은 Mac(#2883 가드레일), 전체 workspace 테스트.

## 5. 후속 조치

- #2885 Part B PR2: chunk 경계에 맞춘 약 16K 토큰 간격의 주기적 checkpoint.
- #2885 Part B PR3: snapshot 사이 간격이 가장 작은 것부터 지우는 eviction.
- 기본 2 GB snapshot 예산은 이 모델에서 snapshot 약 6개만 담는다(조건마다 LRU eviction 15회). 기본값 상향이 필요할 수 있다.
- 세션 간 root 공유 여부는 정책 결정으로 남음.
