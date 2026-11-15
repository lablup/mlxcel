# 기술 리포트: PR #3055 - 턴 경계에서 recurrent 상태 체크포인트 저장

**작성일**: 2026-11-15
**상태**: PR 열림 (draft, #3047 해결)
**검토자**: -
**언어**: Rust
**위험도**: 중간

---

## 요약

하이브리드 모델(Qwen 3.5, Qwen3-Next, Qwen4-Exp)은 GatedDeltaNet 상태를 되감을 수 없으므로, 이전 턴에 대한 prompt cache 히트에는 정확히 그 턴에서 찍은 전체 스냅샷이 필요합니다. 에이전트 대화 길이에서 전체 스냅샷은 1~2 GB이고 대부분이 attention KV이며, session-chain supersede 규칙은 대화당 가장 최근 history boundary 하나만 남깁니다. 따라서 1~3턴 전을 수정하는 요청은 system prompt root보다 가까운 항목을 찾지 못해 약 23K 토큰을 다시 prefill했습니다.

이 PR은 모든 history boundary에서 recurrent 및 convolution 상태만 저장하고(체크포인트, replay에서 151 MB~449 MB), 히트 시에는 이 체크포인트와 그 대화의 가장 최근 전체 스냅샷(donor)으로 전체 상태를 재구성합니다. donor를 복원하고, attention KV를 speculative-rollback trim으로 체크포인트 길이까지 잘라낸 뒤, 체크포인트의 recurrent 상태를 설치합니다. 이 재구성은 donor의 앞쪽 행이 체크포인트를 만든 것과 같은 시퀀스들이 계산한 것일 때만 정확하므로, 모든 스냅샷이 lineage를 갖고 lineage가 일치하는 donor 위에서만 체크포인트를 제공합니다. 체크포인트는 자체 바이트 예산, TTL, LRU, orphan 제거를 가집니다.

M3 Ultra live-replies replay에서 1턴 전 수정의 TTFT는 `qwen3.5-4b-4bit`(16 GB 제한)에서 17.1초에서 1.7초로, `qwen3.8-27b-4bit`(32 GB 제한)에서 87.0초에서 7.9초로 줄었습니다. 3턴 전은 16.8초에서 5.2초, 85.6초에서 25.0초입니다. off 쪽은 system root에서 약 23K 토큰을 prefill하므로 peak footprint도 함께 낮아집니다. 또한 체크포인트 경로에서 드러난 Qwen4-Exp QSA 문제도 고칩니다. 잘라낸 뒤에는, 더 짧은 prefix를 직접 prefill했다면 pooling되지 않았을 블록 요약을 버려야 합니다.

---

## 1. 문제 정의

### 1.1 배경

Transformer attention 상태는 위치 기반입니다. 어떤 prefix의 KV 행은 그 prefix를 공유하는 더 긴 시퀀스의 KV 앞부분과 같으므로 캐시를 더 짧은 prefix로 trim할 수 있습니다. GatedDeltaNet 레이어는 소비한 모든 토큰을 고정 크기 상태에 누적하므로 trim할 수 없습니다. 그래서 하이브리드 모델의 prompt cache는 history boundary에서 전체 상태 스냅샷(KV와 recurrent 상태)을 저장하고, 히트에는 일치한 prefix 길이의 정확한 스냅샷이 필요합니다.

store의 session-chain supersede 규칙은 대화당 가장 최근 history-boundary 스냅샷 하나만 유지합니다(#3035, #3037이 `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN`으로 하나 더 유지하는 옵션을 추가했습니다). 28K 토큰에서 스냅샷은 1~1.8 GB이므로 턴마다 하나씩 유지하면 기본 예산에 맞지 않습니다.

### 1.2 기존 문제점

- 긴 대화의 N-1, N-2, N-3턴을 수정하는 요청은 수정 지점에 전체 스냅샷이 없어 system prompt root로 되돌아갑니다(4b replay에서 캐시 토큰 4,706개, 27b에서 4,744개).
- 비용이 큽니다. 수정된 턴이 약 23K 토큰을 prefill하므로 A9 TTFT는 4b에서 17초, 27b에서 87초이고 peak footprint는 25.7 GB, 51.7 GB입니다.
- 스냅샷 바이트의 대부분은 같은 대화의 더 긴 스냅샷에 이미 있는 데이터(앞쪽 KV 행)입니다. 대체할 수 없는 것은 recurrent 상태뿐입니다.

### 1.3 위험성

| 위험 | 완화 |
|------|------|
| 다른 prefill 분할로 계산된 KV 행과 체크포인트의 recurrent 상태를 섞으면 어떤 단일 실행도 만들지 않는 상태가 됨 | lineage 검사: donor가 체크포인트 길이 범위에서 체크포인트와 같은 생산자를 공유해야 함 |
| 대화당 추가 상주 메모리 | 별도 바이트 예산(기본 체크포인트 12개, 가용 메모리의 1/8 상한), `0`이면 비활성 |
| 세션 또는 템플릿 간 누출 | 체크포인트와 donor 조회 모두 같은 `SessionlessBucketKey`와 스냅샷 조회의 세션 규칙을 요구 |
| 자른 뒤 Qwen4-Exp QSA 인덱스가 직접 prefill과 달라지는 문제 | `forget_unpooled_blocks`, 그리고 네 가지 QSA 구간에 대한 비트 단위 테스트 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

격리는 유지됩니다. `best_state_checkpoint`와 `donor_for`는 요청의 `SessionlessBucketKey`(모델, LoRA, 템플릿, 멀티모달 digest, RoPE 구성)와 스냅샷 조회와 같은 세션 규칙을 요구하고, 토큰 prefix 포함 여부와 lineage 일치도 검사합니다. 시퀀스 id는 단조 증가 카운터에서 나오므로 lineage가 충돌하지 않습니다. 멀티모달 요청은 boundary 스냅샷을 저장하지 않고 다른 digest 버킷을 쓰므로 체크포인트에 도달하지 않습니다. 리뷰에서 체크포인트 분기가 `require_whole_entry` 멀티모달 게이트를 건너뛴다는 점이 지적되었습니다. 현재 그 경로에 도달하는 코드는 없으며, 가드를 추가하면 방어 심층화가 됩니다.

### 2.2 성능 관점

- 삽입 비용: 체크포인트는 `ModelStateSnapshot::subset`으로 boundary 스냅샷에서 만들어지며 원본 텐서의 버퍼를 공유합니다. 삽입 시 복사는 없습니다.
- 조회 비용: 체크포인트에 대한 read lock 한 번, 이후 카운터 갱신과 touch를 위한 write lock 한 번입니다.
- 메모리: donor 스냅샷이 대체된 뒤에는 체크포인트의 recurrent 텐서가 단독으로 남습니다. 체크포인트 예산으로 제한됩니다(replay 종료 시 4b에서 6개 항목 151 MB, 27b에서 449 MB).
- 삽입 시 orphan 제거는 store write lock 아래에서 O(체크포인트 x 같은 세션의 스냅샷 x 프롬프트 길이)입니다(8절 참고).

### 2.3 호환성/의존성 관점

- 새 의존성은 없습니다. 새 환경 변수 `MLXCEL_PROMPT_CACHE_STATE_CHECKPOINT_CAPACITY_BYTES`는 `docs/environment-variables.md`와 `docs/CONTINUOUS_BATCHING.md`에 문서화되었고, `/v1/cache/stats`에 `state_checkpoint_*` 카운터가 추가됩니다.
- `LanguageModel`에 기본 구현이 있는 trait 메서드 두 개가 추가됩니다. `state_checkpoint_from_snapshot`(기본 `None`)과 `restore_sequence_state_from_checkpoint`(기본 `Err`)로, 다른 모델 계열은 전체 스냅샷만 계속 사용합니다. VL 래퍼(`qwen3_5_vl`, `qwen4_exp_vl`, `minicpmv4_6_vl`)와 `LanguageModelRef`가 이를 전달합니다.
- `ModelSnapshotEntry`에 `lineage` 필드가 추가되며 기본값은 빈 값(아무것과도 일치하지 않음)입니다.

### 2.4 코드 품질 관점

- Qwen4-Exp의 linear 레이어 복원 로직을 `restore_linear_layer`로 추출해 전체 스냅샷 경로와 체크포인트 경로가 공유합니다.
- 자르기는 새 truncation 경로를 만들지 않고 `KVCache::trim`과 Qwen4-Exp rollback trim을 재사용합니다.
- `state_checkpoints.rs`의 `same_session`은 `snapshot_lookup`의 인라인 세션 비교와 중복됩니다(LOW, 리뷰 지적).

---

## 3. 기술적 선택과 그 이유

### 3.1 전체 스냅샷을 늘리지 않고 체크포인트와 donor 사용

| 선택지 | 결과 |
|--------|------|
| 스냅샷 예산을 늘리거나 `KEEP_PREVIOUS_TURN` 사용 | 유지하는 턴당 1 GB 수준. Flash-Next에서 전체 스냅샷 2,946 MB 대 (1,191 MB + 체크포인트 662 MB) |
| 스냅샷마다 copy-on-write KV로 턴별 저장 | 스냅샷용 paged 또는 공유 KV가 필요해 변경 범위가 훨씬 큼 |
| recurrent 상태만 저장하고 KV는 최신 스냅샷에서 빌려옴 | 채택: 턴당 수십~수백 MB, 기존 trim과 복원 코드를 재사용 |

### 3.2 토큰 일치를 믿지 않고 lineage 사용

토큰이 같다고 행이 같은 것은 아닙니다. chained hit은 cold 실행과 다른 조각으로 히스토리를 forward하며, 이는 bf16 값을 몇 ULP 움직입니다(기존 partition effect). 다른 분할에서 나온 앞쪽 행의 donor를 잘라 체크포인트의 recurrent 상태와 합치면, 어떤 단일 실행도 만들지 않는 상태가 됩니다. `SnapshotLineage`는 각 스냅샷 행에 대해 `(end, 시퀀스 id)` span을 기록합니다. 복원은 행을 그대로 복사하므로, 스냅샷을 채택한 시퀀스는 그 스냅샷의 lineage를 일치 길이로 잘라 물려받고, 직접 계산한 행은 자신의 id로 기록합니다. `shares_prefix(other, len)`은 두 lineage가 모두 `len`행을 덮고 `len`으로 자른 값이 같을 때만 참이며, 빈 lineage는 절대 일치하지 않습니다. lineage는 채택, warm-up, 대기열 re-prefill, handoff를 거쳐 시퀀스를 따라갑니다.

### 3.3 별도 예산, TTL, LRU, orphan 제거

체크포인트는 별도 맵에 있으므로 전체 스냅샷이 체크포인트를 밀어내지 않고 그 반대도 마찬가지입니다. 재구성할 donor가 없는 체크포인트는 쓸모가 없으므로, 체크포인트를 삽입할 때마다 어떤 저장된 스냅샷도 재구성할 수 없는 체크포인트를 제거합니다(체크포인트와 정확히 같은 토큰의 스냅샷은 후속 스냅샷이 donor가 되므로 인정). 기본 예산은 모델 recurrent 상태 체크포인트 12개(float32 상태, 스트림 dtype의 윈도, `qwen4_exp`는 PLE 윈도 포함)이며, 모델 적재 후 남은 메모리의 1/8로 제한됩니다. 이 예산은 추가분이므로 스냅샷 store의 1/4의 절반입니다.

### 3.4 조회 우선순위

체크포인트는 어떤 전체 또는 부분 스냅샷보다 더 멀리 도달할 때만 제공되므로 전체 스냅샷이 항상 우선합니다. 또한 요청 전체보다 짧아야 합니다(`len < tokens.len()`). prefill이 logits를 위해 마지막 프롬프트 토큰 이상을 다시 실행하고, recurrent 상태는 한 토큰조차 되감을 수 없기 때문입니다. 체크포인트 복원이 실패하면 cold prefill로 돌아가며 더 짧은 전체 스냅샷 히트를 다시 시도하지 않습니다(LOW, 리뷰 지적).

### 3.5 Qwen4-Exp: 자른 뒤 pooling되지 않은 QSA 블록 제거

QSA indexer는 쿼리가 `block_topk`보다 많은 완성 블록을 볼 때에야 블록 요약을 pooling합니다. `index.trim`은 더 이상 완성되지 않은 블록의 요약을 제거하지만, 긴 길이로 prefill된 donor에는 더 짧은 prefix를 직접 prefill했다면 계산되지 않았을 pooling 요약이 남아 있을 수 있습니다. `QsaAttention::forget_unpooled_blocks`는 `index.len() / compress_ratio <= block_topk`이면 `block_keys = None`으로 설정해, 잘린 캐시가 짧은 prefix의 직접 prefill 결과와 같아지게 합니다. `DecoderLayer::trim_cache_to_prefill_state`가 rollback trim과 이 단계를 묶습니다. 이 단계가 없으면 fixture 테스트의 네 가지 QSA 구간이 직접 prefill과 일치하지 않습니다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
Boundary 스냅샷 삽입 (scheduler):
  snapshot --subset(recurrent 텐서, 버퍼 공유)--> checkpoint
  lineage = adopted_lineage.extended(len, 이 시퀀스 id)
  store.insert_snapshot(snapshot, lineage); store.insert_state_checkpoint(checkpoint, lineage)

조회 (store.snapshot_lookup):
  최선의 전체/부분 스냅샷 -> reach
  best_state_checkpoint(요청의 prefix, len > reach, len < 요청 길이) + donor_for(lineage가 prefix 공유)
  -> CheckpointHit { checkpoint, donor, matched_len }

채택 (scheduler.adopt_state_checkpoint):
  model.restore_sequence_state_from_checkpoint(seq, checkpoint, donor)
    attention 레이어: donor 복원, matched_len까지 trim (+ QSA 인덱스, pooling되지 않은 블록 제거)
    linear 레이어: 체크포인트 상태 설치
  adopted_lineage[seq] = checkpoint.lineage
```

### 4.2 주요 코드 변경

**모델** (`qwen3_5.rs`, `qwen3_next.rs`, `qwen4_exp.rs`, `qwen4_exp_snapshot.rs`, `qwen4_exp_qsa.rs`, VL 래퍼): `Qwen3NextCache::is_checkpoint_tensor`는 이름에 `.linear.`가 들어간 텐서를 유지하고, Qwen4-Exp는 `.kv.`와 `.qsa.` 텐서를 제외한 모든 것을 유지합니다(레이어 테이블, PLE 윈도, n-gram 컨텍스트 포함). `restore_from_checkpoint`와 `checkpoint_caches`는 계열, 길이, 레이어 구성을 검증하고 아무것도 설치하기 전에 `Err`를 반환합니다.

**코어** (`generate.rs`, `language_model_ref.rs`): `ModelStateSnapshot::subset`과 trait 메서드 두 개.

**Store** (`prompt_cache/store.rs`, `state_checkpoints.rs`, `lineage.rs`, `entry.rs`, `policy.rs`, `snapshot_sizing.rs`): `Inner`의 체크포인트 맵과 카운터, `insert_state_checkpoint`, `best_state_checkpoint`, `donor_for`, `drop_orphan_state_checkpoints`, TTL sweep, LRU 상한, `SnapshotLookupOutcome::CheckpointHit`, `clear()` 처리, 설정 필드, 모델 인식 크기 산정(`recommend_state_checkpoint_capacity`).

**스케줄러** (`scheduler/prompt_cache.rs`, `queued_adoption.rs`, `handoff.rs`, `admission.rs`, `mod.rs`): `adopt_state_checkpoint`, `Boundary` 출처에 대한 `insert_model_state_snapshot`의 체크포인트 생성, lineage 전달, `startup.rs`의 시작 시 연결(운영자 override 우선, 없으면 모델 인식 기본값).

### 4.3 데이터 모델 변경

`ModelSnapshotEntry.lineage`, `PromptCacheConfig.state_checkpoint_capacity_bytes`, 그리고 `PromptCacheStats`의 새 필드 다섯 개(`state_checkpoint_entries`, `_bytes`, `_inserts`, `_hits`, `_evictions`)가 추가됩니다. 영속되는 데이터는 없습니다.

---

## 5. 학습 포인트

### 5.1 되감을 수 있는 것과 없는 것을 분리

**개념:** 하이브리드 모델의 상태는 두 종류입니다. 위치 기반 행(KV, QSA 인덱스)은 더 긴 상태의 prefix이므로 빌려올 수 있고, recurrent 요약은 그럴 수 없습니다. 턴마다 후자만 저장하면 턴당 기가바이트 문제가 수백 메가바이트 문제로 줄어듭니다.

**이 PR에서의 적용:** 체크포인트가 되감을 수 없는 부분집합이고, 나머지는 donor가 제공합니다.

### 5.2 같은 토큰이 같은 상태는 아니다

bf16에서는 히스토리를 prefill에 어떻게 나눠 넣었는지에 따라 행이 몇 ULP 달라집니다. 서로 다른 실행의 상태를 이어붙이는 방식은 각 부분을 어떤 계산이 만들었는지 추적해야 합니다. lineage는 자르고 비교하기 쉬운 span 목록으로 이를 수행합니다.

### 5.3 partition effect가 `--verify` 불일치를 설명한다

`qwen3.5-4b`의 off 쪽은 이미 cold 실행 대비 A6, A8을 놓치고, on 쪽은 추가로 A9(그리고 이어서 A10)를 놓칩니다. 25,625 토큰에서 전체 스냅샷을 유지한 chained 실행(`MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN=1`, 128G)은 체크포인트 실행과 A9, A10을 포함한 16개 completion이 바이트 단위로 같습니다. 즉 체크포인트는 대체한 스냅샷을 그대로 재현하며, cold와의 차이는 문서화된 chained-hit partition effect입니다. 히트 지점이 옮겨가면 이 효과가 나타나는 턴도 옮겨갑니다. tiny-fixture 테스트도 같은 동일성을 비트 단위로 고정합니다.

---

## 6. 추가 학습 리소스

### 핵심 키워드
| 키워드 | 설명 | 관련성 |
|--------|------|--------|
| GatedDeltaNet | 고정 크기 recurrent 상태를 가진 선형 attention 레이어 | 되감을 수 없는 상태 |
| Donor | 대화의 가장 최근 전체 스냅샷 | 체크포인트 prefix의 attention 행을 제공 |
| Lineage | 각 행을 계산한 시퀀스의 span별 기록 | 서로 다른 분할이 섞이는 것을 방지 |
| Partition effect | prefill 분할이 다르면 bf16 logits가 몇 ULP 달라짐 | cold 대비 `--verify` 불일치의 원인 |
| QSA | Qwen4-Exp의 블록 pooling 희소 attention 인덱스 | 자른 뒤 `forget_unpooled_blocks` 필요 |

### 관련 PR/이슈
- #3047: 이 기능 (이 PR이 해결)
- #3035, #3037: keep-previous-turn 복원 지점
- #3040: 하이브리드 캐시 히트의 출력 동일성 기준
- #2883: 스냅샷 증가 guardrail (체크포인트에는 적용되지 않음, 8절 참고)

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|------|-----|
| 변경된 파일 | 86개 (JSON 제외 45개, replay 결과 파일 41개) |
| 추가된 줄 | +9,176 (replay 원본 JSON 제외 +2,434) |
| 삭제된 줄 | -70 |
| 새 테스트 파일 | 4개 (`qwen3_5_checkpoint_tests`, `qwen4_exp_checkpoint_tests`, `scheduler_state_checkpoint_tests`, `state_checkpoints_tests`), 그 외 sizing, lineage, policy 테스트 |

### 카테고리별 변경

| 카테고리 | 요약 |
|----------|------|
| 기능 | 체크포인트 생성, lineage, donor 재구성, 자체 예산, 조회 우선순위, 스케줄러 채택 |
| 버그 수정 | QSA 자르기를 위한 Qwen4-Exp `forget_unpooled_blocks` |
| 문서 | `CONTINUOUS_BATCHING.md`, `environment-variables.md`, replay 결과 |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|--------|
| `5a22cf3` | feat | checkpoint recurrent state at turn boundaries |
| `6a21ac2` | docs | add the turn-boundary checkpoint replay results |

---

## 8. 후속 조치

### 완료 필요
- [ ] (MEDIUM) 체크포인트 바이트가 `store.bytes()`와 `store.len()`(`store.rs`)에 포함되지 않아 prompt-cache gauge가 이를 과소 보고하며, #2883 증가 guardrail은 스냅샷 삽입만 다루고 `insert_state_checkpoint`는 다루지 않습니다. 메모리는 체크포인트 예산(기본은 가용 메모리의 1/8)으로 제한되고 `clear()`로 해제되지만, gauge를 보는 운영자에게는 보이지 않습니다.
- [ ] (MEDIUM) `drop_orphan_state_checkpoints`는 삽입마다 store write lock을 잡은 채 모든 체크포인트를 모든 스냅샷과 비교합니다(토큰 `starts_with`와 쌍마다 `clipped` 할당 두 번). 항목 12개 정도에서는 저렴하지만, env override로 예산을 키우면 체크포인트 x 같은 세션 스냅샷 x 프롬프트 길이로 증가합니다.

### 모니터링 필요
- 긴 에이전트 대화에서 `/v1/cache/stats`의 `state_checkpoint_hits`, `_evictions`, `_bytes`.
- 한 서버에서 많은 대화를 처리할 때 기본값 12개 체크포인트 대비 히트율.

### 향후 개선 사항
- (LOW) orphan은 체크포인트 삽입 시에만 제거됩니다. 스냅샷 LRU나 TTL이 donor를 제거하면 의존하는 체크포인트는 자체 TTL이나 LRU까지 남습니다.
- (LOW) `adopt_state_checkpoint`에 `require_whole_entry` 가드 추가.
- (LOW) 기존 문제: 저장된 스냅샷이 요청 전체를 덮는 경우(동일 프롬프트 재생성) `reach`가 프롬프트 길이와 같아져 하이브리드는 cold prefill합니다.
- (LOW) env override는 가용 메모리로 제한되지 않으며, 이는 문서화된 운영자 의미와 같습니다.
- (LOW) `same_session`을 `snapshot_lookup`과 공유.

---

## 부록

### A. 테스트 결과

- tiny-fixture 테스트: 재구성된 상태는 qwen3_5와 qwen4_exp(네 가지 QSA 구간)에서 더 짧은 prefix의 직접 prefill과 비트 단위로 같고, 그 뒤의 decode logits와 그 위에 prefill한 suffix도 같습니다. 일치하지 않는 donor는 거부됩니다.
- 스케줄러 테스트: 2턴 전 수정은 체크포인트를 채택하고 전체 스냅샷을 가진 reference와 같게 decode하며, 예산이 없으면 처음부터 시작합니다.
- 대상 lib 테스트(prompt_cache, scheduler, qwen4_exp, qwen3_5, qwen3_next)와 `cargo clippy --lib --tests -D warnings`는 PR 기준으로 통과합니다. 이 리포트를 작성하면서 테스트를 실행하지는 않았습니다.

### B. Replay 결과 (M3 Ultra)

`--turn-tokens 3072 --edit-turn 9 --live-replies`, off = 체크포인트 예산 0(main의 동작), 셀당 2~3회 반복, 한 번의 GPU-lock 보유 안에서 off/on을 교차 실행. 빌드는 `ab45ac59`의 main에 #3047을 더한 것입니다.

| 모델, 제한, 수정 | A9 캐시 토큰 off / on | A9 TTFT off / on | 체크포인트 바이트 | Peak footprint off / on |
|---|---|---|---|---|
| qwen3.5-4b, 16G, 1턴 전 | 4,706 / 25,625 | 17.05-17.38 / 1.70-1.71 s | 151 MB | 25.7 / 13.5 GB |
| qwen3.5-4b, 16G, 3턴 전 | 4,706 / 19,895 | 16.69-16.86 / 5.22-5.25 s | 151 MB | 27.2 / 17.3 GB |
| qwen3.8-27b, 32G, 1턴 전 | 4,744 / 25,885 | 87.04-87.07 / 7.89-7.98 s | 449 MB | 51.7 / 33.5 GB |
| qwen3.8-27b, 32G, 3턴 전 | 4,744 / 20,081 | 85.58-85.64 / 24.96-24.98 s | 449 MB | 52.7 / 36.4 GB |

전체 스냅샷 바이트는 변하지 않았습니다(4b 1,059 MB, 27b 2,201 MB). replay 전체 TTFT 합계는 4b가 43.4초에서 28.0초(1턴 전), 43.0초에서 31.6초(3턴 전), 27b가 219.4초에서 140.2초, 218.0초에서 157.3초입니다. peak footprint는 `--verify` 없는 실행에서 가져왔고 모델 가중치와 prefill activation을 포함합니다. 시작 시 load average는 1.2~7.7이었고 4b 실행 두 번은 21.3과 14.0이었으며, 반복 간 편차는 전체 TTFT 기준 0.4초 이내였습니다.

cold 실행 대비 `--verify` 동일성:

| 모델, 수정 | off cold 일치 | on cold 일치 |
|---|---|---|
| 4b, 1턴 전 | 14/16 (A6, A8 불일치) | 12/16 (A6, A8, A9, A10) |
| 4b, 3턴 전 | 14/16 (A6, A8) | 13/16 (A6, A8, A9) |
| 27b, 1턴 및 3턴 전 | 14/16 (B2, B3) | 14/16 (B2, B3) |
| Flash-Next, 1턴 전 | 해당 없음 | 16/16 |

Flash-Next에서는 체크포인트 쪽과 keep-previous-turn 쪽 모두 cold 실행 16개와 일치합니다. 체크포인트 쪽은 전체 스냅샷 1,191 MB와 체크포인트 662 MB를 보유하고, keep-previous-turn 쪽은 전체 스냅샷 2,946 MB를 보유합니다. 4b의 on 쪽 불일치는 전체 스냅샷 실행과 바이트 단위로 같다는 것이 확인된 partition effect입니다(5.3절). 측정하지 않은 것: Flash-Next의 3턴 전, 그리고 이 세 모델 외의 모델.

### C. 참고 자료
- 이슈 #3047
- `docs/benchmark_results/hybrid-turn-state-checkpoints-replay-m3ultra-2026-11-15.md`
- `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`
