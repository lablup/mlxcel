# 기술 보고서: PR #3044 - fix(prompt-cache): chat 턴을 history boundary에서 재개

**날짜**: 2026-11-14
**상태**: 완료
**언어**: Rust
**위험도**: 중간

---

## 요약

snapshot 전용 계열(Qwen 3.x 하이브리드, Gemma 4, SSM 하이브리드)에서 prompt cache는 생성이 끝난 시점에 `prompt + reply`를 키로 completion snapshot을 저장했습니다. 이 snapshot의 reply 행은 한 토큰씩 진행하는 decode 단계가 쓴 것이고, cold prefill은 이런 분할을 쓰지 않으므로, 이를 복원하면 다음 턴의 cold 실행이 재현되지 않았습니다. Qwen3.8-Flash-Next에서 live reply로 재생하면 `--verify` 요청 16개 중 13개가 cold 실행과 달랐습니다(#3040). 이제 history boundary가 있는 chat 턴은 completion snapshot을 아예 저장하지 않으므로 다음 턴은 boundary에서 재개하고, boundary가 없는 요청은 prefill이 끝나는 시점의 prompt 끝 상태를 저장합니다. 재생 결과는 16개 중 16개가 일치했고, 비용은 턴마다 33~36개 토큰의 재prefill(평균 TTFT +0.054초, 약 2%)입니다. `MLXCEL_PROMPT_CACHE_DECODED_TAIL=1`로 이전 동작을 되돌릴 수 있습니다.

---

## 1. 문제 정의

### 1.1 배경

snapshot 계열은 recurrent 또는 linear-attention 상태를 모델 소유 버퍼에 두기 때문에 요청 간에 KV 블록을 공유할 수 없습니다. 대신 prompt cache가 정해진 지점에서 상태를 복사해 두었다가, 저장된 토큰 벡터를 확장하는 이후 요청에 복원합니다. #1143 이후 chat 턴은 snapshot을 두 개 저장합니다. history boundary(generation prompt 없이 렌더링한 대화, prefill 도중 캡처)와 completion snapshot(종료 시 기증)입니다. #2974는 hit이 cold 실행을 재현하려면 복원 지점 이후에 cold prefill이 forward하는 조각과 정확히 같은 조각을 forward해야 한다는 점을 밝혔습니다.

### 1.2 기존 문제점

- `--live-replies`(모델 자신의 답을 수정 없이 다시 넣음)에서는 매 턴이 이전 턴의 completion snapshot을 채택했습니다. 저장된 가장 긴 prefix였기 때문입니다.
- 그 snapshot의 꼬리는 decode가 쓴 것입니다. 모델 커널은 행 수에 따라 동작이 달라서, 한 행씩 쓴 상태는 같은 토큰을 prefill로 쓴 상태와 바이트 단위로 같지 않습니다. `qwen4_exp_decode_tail_tests`가 작은 fixture에서 이를 고정합니다. decode가 쓴 상태는 prompt boundary에서 나눈 prefill과도, 한 번에 처리한 prefill과도 일치하지 않고, prompt boundary 상태를 복원한 뒤 reply를 prefill하면 cold 분할과 비트 단위로 같습니다.
- 차이는 반올림 수준이지만, greedy decoding은 두 후보가 bfloat16 몇 단위 안에 있으면 바로 뒤집히므로 재생 요청 16개 중 13개가 cold 실행과 다른 텍스트를 냈습니다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 캐시된 턴이 빈 캐시의 같은 요청과 다르게 답함 | 재현 불가능한 출력, `--verify` 게이트 실패 | 수정 전 높음(16개 중 13개) |
| 매 턴 reply 재prefill로 TTFT 증가 | chat 지연 | 확실, 약 2%로 측정 |
| `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN`(#3037)이 restore source를 잃음 | 편집이 system-prompt root로 후퇴 | 낮음: restore source는 origin과 무관하게 digest로 추적 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

새로운 입력 경로는 없습니다. 새 환경 변수는 `OnceLock`으로 한 번 읽는 boolean이며 기존 `parse_switch`로 해석합니다.

### 2.2 성능 관점

- 다음 턴은 generation prompt와 reply를 다시 prefill합니다. 재생에서 턴당 33~36 토큰, 이전 턴을 채택하는 수정 없는 12개 턴에서 TTFT +0.022~+0.098초였습니다.
- boundary가 없는 요청의 prompt 끝 복사는 첫 토큰을 내보낸 뒤 `finish_prefill`에서 실행되므로 TTFT에 더해지지 않고, 생성 종료 시점의 복사를 일대일로 대체합니다.
- boundary가 있는 chat 턴은 턴당 상태 복사가 두 번에서 한 번으로 줄고, 대화 하나가 보유하는 snapshot도 하나 줄어 snapshot 예산에 여유가 생깁니다.

### 2.3 호환성/의존성 관점

- `prompt + reply` 정확 일치 항목에 의존하던 raw completion은 이제 prompt 끝을 복원하고 reply를 prefill합니다.
- speculative burst는 classic 경로로 prefill을 끝내지 않으므로 생성 종료 시점의 snapshot을 그대로 유지합니다.
- prefill 안에서 끝나는 요청(즉시 EOS, 한 토큰 제한)은 이전처럼 종료 시 기증하며, 그 상태에는 prompt만 들어 있습니다.

### 2.4 코드 품질 관점

결정 로직은 새 모듈 `scheduler/prompt_end_snapshot.rs`에 있고, `HashSet<SequenceId>`(`prompt_cache_prompt_end_stored`)가 `donate_finished_sequence_cache`에 종료 시점 복사를 건너뛰라고 알립니다. 이 집합은 donate와 `release_sequence_caches` 양쪽에서 정리되므로 중단되거나 오류로 끝난 sequence도 항목을 남기지 않습니다.

---

## 3. 기술적 선택과 그 이유

### 3.1 prompt를 키로 쓰는 대신 history boundary에서 재개

**맥락**: 첫 커밋은 completion snapshot을 prompt 끝에서 prompt만을 키로 저장해 decode가 쓴 reply를 제거했습니다.

**검토한 선택지**:
1. prompt 끝 snapshot: 재생에서 여전히 16개 중 3개.
2. boundary가 있는 chat 턴은 completion snapshot을 저장하지 않음: 16개 중 16개.
3. 생성 종료 시점 snapshot을 유지하고 불일치를 감수(이전 기본값).

**결정**: 기본값은 선택지 2, `MLXCEL_PROMPT_CACHE_DECODED_TAIL=1`에서는 선택지 3.

**이유**: history boundary와 prompt 끝 사이의 행은 generation prompt이고, 다섯 행짜리 마지막 조각으로 forward되어 small-row 커널을 탑니다. 다음 턴의 cold 실행은 이 토큰들을 history 구간 안에서 forward하므로, 이를 포함한 상태는 어떤 cold 실행도 재현하지 못합니다. 다음 턴의 cold plan과 같은 지점에서 조각이 끝나는 곳은 boundary뿐입니다.

### 3.2 생성 종료가 아니라 prefill 종료 시점에 결정

`finish_prefill`이 sequence를 decode로 넘기는 순간에만 상태가 prefill로만 쓰여 있습니다. 여기서 결정하고(그 결정을 종료 시점이 참조하는 집합에 기록) 나중에 상태를 되감을 필요가 없으며, 이 호출에 도달하지 않는 speculative 경로는 건드리지 않습니다.

### 3.3 boundary를 캡처했는지가 아니라 boundary가 있는지로 판단

리뷰 중 조건을 `history_boundary_split(seq).is_some()`에서 `ctx.history_prefix_tokens.is_some()`로 바꿨습니다. 재생성(regenerate)된 턴은 같은 턴의 이전 요청이 저장한 boundary snapshot을 채택하므로 `boundary_capture_applies`가 거절하고(채택한 prefix가 이미 boundary에 닿음), 원래 조건은 prompt 끝 저장으로 넘어갔습니다. 다섯 행짜리 generation prompt 조각을 포함한 그 항목은 다음 턴에서 가장 긴 일치가 되어 불일치를 다시 들여옵니다. `history_prefix_tokens`는 `resolve_history_boundary`가 prompt의 진 prefix로, 또는 boundary 기능이 꺼졌거나 multimodal 요청이면 `None`으로 이미 정규화하므로 올바른 판단 기준입니다. scheduler 테스트는 이제 턴 2를 재생성하고 항목이 늘지 않음을 확인합니다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
finish_prefill (classic 경로)
  ├─ prefill 안에서 종료 → donate_finished_sequence_cache (변경 없음)
  └─ decode로 진행 → insert_prompt_end_snapshot
        ├─ DECODED_TAIL=1 또는 snapshot 계열 아님 → 아무것도 안 함 (종료 시 기존 경로)
        ├─ prompt_cache_prompt_end_stored에 seq 기록
        ├─ 요청에 history boundary 있음 → 저장 안 함
        └─ 그 외 → insert_model_state_snapshot(prompt, Completion)
donate_finished_sequence_cache
  └─ 기록된 seq → snapshot 계열 복사 생략
release_sequence_caches → 집합에서 seq 제거
```

### 4.2 주요 코드 변경

- `src/server/batch/scheduler/prompt_end_snapshot.rs`(신규): `insert_prompt_end_snapshot`.
- `src/server/batch/scheduler/prefill.rs`: `Decoding` 전환 전에 호출.
- `src/server/batch/scheduler/prompt_cache.rs`: `donate_finished_sequence_cache`가 먼저 기록을 제거하고, 기록이 있었으면 복사를 생략.
- `src/server/prompt_cache/decoded_tail.rs`(신규): thread-local 테스트 override가 있는 `decoded_tail_snapshot_enabled()`.
- `src/models/qwen4_exp_decode_tail_tests.rs`(신규): 모델 수준 근거.
- `src/server/batch/scheduler_completion_snapshot_tests.rs`: prompt 끝 항목만 저장, opt-out은 이전 항목 유지, chat 턴은 boundary에서 재개, 재생성은 추가 항목을 저장하지 않음, hit이 cold 실행과 같은 결과를 decode.

### 4.3 데이터 모델 변경

`BatchScheduler`에 `prompt_cache_prompt_end_stored: HashSet<SequenceId>`가 추가됩니다. store나 항목 형식은 바뀌지 않고, completion 항목은 계속 `SnapshotOrigin::Completion`입니다.

---

## 5. 학습 포인트

### 5.1 decode와 prefill은 다른 바이트를 쓴다

이 커널들에서 토큰을 decode로 쓰는 것과 같은 토큰을 prefill로 쓰는 것은 비트 수준에서 서로 바꿔 쓸 수 없습니다. cold 실행과의 동일성을 원하는 캐시는 모든 행이 cold 실행과 같은 분할로 쓰인 상태만 재사용할 수 있습니다.

### 5.2 generation prompt도 분할의 일부다

reply만 제거해서는 부족했습니다. generation prompt 스캐폴드도 다음 턴의 plan이 나누는 지점 너머에 있기 때문입니다. 중간 빌드의 측정(여전히 16개 중 3개)이 이 위치를 찾아냈습니다.

### 5.3 캡처와 존재는 다른 조건이다

"이 요청이 boundary를 캡처했다"와 "이 요청에 boundary가 있다"는 boundary가 이미 존재하는 재생성에서 정확히 갈립니다. 무엇을 저장하지 않을지 정하는 조건은 후자를 물어야 합니다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

- **History boundary**: chat history가 끝나고 generation prompt가 시작하는 토큰 위치(#1143).
- **Partition effect**: prompt를 forward 단위로 어떻게 나누느냐에 따라 모델 상태 바이트가 달라지는 현상(#2974).
- **Restore source**: 세션의 최신 턴이 채택한 snapshot, `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN`이 보존(#3035, #3037).

### 관련 PR/이슈

- #3040: 이 PR이 닫는 이슈.
- #2974, #2975: partition effect와 연쇄 복원.
- #3035, #3037: 이전 턴 복원 지점.
- #1143, #1754: history boundary와 completion snapshot 집계.

---

## 7. 변경 요약

### 통계

약 18개 파일: 소스 7개, 테스트 3개, 문서 2개, 벤치마크 보고서와 원시 JSON 3개.

### 카테고리별 변경

- Scheduler: prompt 끝 결정, donate 생략, 기록 정리.
- Prompt cache: opt-out 스위치.
- 테스트: 모델 수준 decode 꼬리 근거, scheduler 수준 저장 및 cold 동일성 테스트.
- 문서: `CONTINUOUS_BATCHING.md`, `environment-variables.md`, 재생 보고서.

### 관련 커밋

- `fix(prompt-cache): store completion snapshots at the prompt boundary`
- `fix(prompt-cache): resume chat turns from the history boundary`
- `origin/main` 병합
- `fix(prompt-cache): skip the prompt-end snapshot on a regenerated turn`

---

## 8. 후속 조치

### 모니터링 필요

- reply가 긴 장기 multi-turn chat의 TTFT. reply 재prefill 비용이 32 토큰으로 측정한 재생보다 큽니다.

### 향후 개선 사항

- prefill 안에서 끝나는 chat 턴(즉시 EOS 또는 한 토큰 제한)은 여전히 generation prompt 조각을 포함한 prompt 끝 항목을 기증합니다. 드문 경우라 그대로 두었습니다.
- speculative burst는 여전히 decode가 쓴 생성 종료 snapshot을 저장합니다. 같은 방식으로 cold 실행과 갈라지지만 이번에 측정하지 않았습니다.

---

## 부록

### A. 테스트 결과

- `models::qwen4_exp::decode_tail_tests`, `models::qwen4_exp::snapshot_tests`
- `server::batch::scheduler_completion_snapshot_tests`
- `server::prompt_cache`

### B. 성능 벤치마크

| 구성 | cold 일치 | 수정 없는 턴 TTFT | TTFT 합계 |
|---|---|---|---|
| `MLXCEL_PROMPT_CACHE_DECODED_TAIL=1` | 16개 중 3개 | 2.62~3.08초 | 70.00초 |
| prompt 끝 snapshot(중간 빌드) | 16개 중 3개 | 2.67~3.12초 | 70.61초 |
| 기본값: boundary에서 재개 | 16개 중 16개 | 2.62~3.13초 | 70.54초 |

M3 Ultra, `Qwen3.8-Flash-Next-OptiQ-2bit`, `--turn-tokens 3072 --edit-turn 9 --edit-target 8 --live-replies --verify`, `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1`.

### C. 참고 자료

- `docs/benchmark_results/qwen4-exp-completion-snapshot-replay-m3ultra-2026-11-14.md`
- `docs/CONTINUOUS_BATCHING.md`, one-prefill plan 절
