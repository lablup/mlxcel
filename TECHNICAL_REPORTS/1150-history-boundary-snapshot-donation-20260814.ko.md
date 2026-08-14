# 기술 보고서: PR #1150 - feat(server): donate a history-boundary snapshot during prefill

**날짜**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: 중간 (해당 요청의 prefill에 기본 켜짐으로 추가 forward와 상태 복사가 붙음. snapshot-reuse 계열로 게이트했고 킬 스위치를 제공하며 dense-KV 계열은 불변임을 검증)

---

## 요약

snapshot-only 계열(`supports_snapshot_reuse()`를 보고하는 13개 계열)은 저장된 토큰 벡터에 대한 exact-prefix 매치로만 프롬프트 캐시를 재사용한다. 생성 종료 시 기증되는 벡터는 `prompt + generated`인데, 에픽 #1148은 이 벡터가 시험한 모든 계열에서 다음 턴의 접두사가 되지 못함을 실측했다. 원인은 서로 독립적인 세 가지이고, 셋 다 히스토리 경계 너머에 있다. 템플릿이 생성 프롬프트에만 스캐폴드를 붙이고, 어시스턴트 턴을 히스토리로 재렌더링할 때 thinking 블록을 떼며, 샘플링된 토큰 열은 자기 텍스트의 정준 토크나이즈가 아니다.

이번 변경은 prefill 중 히스토리 경계에서 두 번째 스냅샷을 찍는다. 키는 `add_generation_prompt = false` 렌더 자체를 토크나이즈한 벡터다. 기증 벡터가 재렌더링된 히스토리 형태를 토크나이즈해 만들어지므로, 구성상 모든 후속 턴의 접두사가 되고 세 발산 클래스를 한꺼번에 무력화한다. `main` 9c154ff3 대비 A/B 실측(qwen3.5-0.8b-4bit): 둘째 턴 `cached_tokens` 0/189에서 150/189로, 셋째 턴 0/214에서 184/214로, `snapshot_hits` 0에서 2로. dense-KV 대조군 llama-3.2-1b는 모든 셀에서 두 팔이 동일했다.

동시에 머지된 형제 PR 위로 리베이스하다가 진짜 교차 이슈 상호작용을 발견했다. #1151의 세션 체인 supersede가 경계 스냅샷을 만들어지자마자 지워 버렸다. 스냅샷에 `SnapshotOrigin`을 달아 생산자별로 체인을 나눠 고쳤다.

---

## 1. 문제 정의

### 1.1 배경

snapshot-only 계열에서 `PromptCacheStore::lookup_snapshot_prefix`가 유일한 재사용 경로이고, 저장 벡터가 들어오는 요청의 exact prefix여야 한다. 2026-08-14 라이브 검증(main 9c154ff3)에서 둘째 턴 `cached_tokens = 0`이 gemma-4-31b(169토큰 프롬프트), 두 thinking 모드의 qwen3.5-4b(123, 177), falcon-h1-tiny(250)에서 재현됐고, 같은 바이너리의 llama-3.2-1b 대조군은 308 중 256을 적중했다. 스토어, 키 구성, adopt 경로는 모두 정상이었다. 실패는 스냅샷 경로가 저장하는 내용에 특정된 문제였다.

### 1.2 기존 문제

- **클래스 (a): 생성 프롬프트 전용 토큰.** gemma-4-31b는 빈 thought 스캐폴드를 생성 프롬프트에만 붙이고 히스토리에는 붙이지 않는다. 토큰 수준 증거로 저장된 139토큰 엔트리가 둘째 턴과 정확히 90개를 공유했고, 발산 토큰 4개를 detokenize하면 그 스캐폴드였다. `enable_thinking=false`의 qwen3.5는 빈 `<think>\n\n</think>\n\n` 주입으로 같은 형태를 보인다.
- **클래스 (b): 히스토리에서 제거되는 thinking.** qwen3.5 기본 모드는 생성 프롬프트에 `<think>\n`을 심지만 어시스턴트 히스토리는 그 블록 없이 재렌더링해서, 어시스턴트 헤더 직후에 벡터가 갈라진다.
- **클래스 (c): 재토크나이즈 드리프트.** falcon-h1-tiny는 평범한 ChatML 템플릿인데도 미스였다. 샘플링된 완료 토큰 120개가 같은 답변 텍스트를 다시 토크나이즈하면 118개가 된다. 샘플링된 열은 정준 토크나이즈가 아니다.
- **공통 성질.** 세 발산 모두 히스토리 경계 이후에서 일어난다. `add_generation_prompt = false`로 렌더링되는 부분은 템플릿의 버릇과 무관하게 턴을 넘어 접두사로 안정적이다. 이 불변식이 이번 변경의 토대다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|---|---|---|
| 13개 계열이 멀티턴 캐시에 전혀 적중하지 못하고 매 턴 전체 prefill을 지불 | 높음 | 확실 (변경 전) |
| 추가 forward나 오프셋 전진이 prefill 경로(청크, 배치, 스펙큘러티브, VLM, 분산)를 오염 | 높음 | 낮음 (진입점 전수 추적, 거부 시 기존 prefill 유지) |
| forward 하나를 둘로 쪼개며 근소 동률에서 greedy 출력이 이동 | 낮음 | 중간 (#203/#325/#326로 문서화된 클래스, 청크 prefill과 캐시 히트에서 이미 현상 유지) |
| dense-KV 배포가 버려질 두 번째 렌더와 인코딩을 지불 | 낮음 | 확실 (수용, 후속 #1153) |

---

## 2. 기술 검토

### 2.1 핵심 불변식: 벡터는 라이브 프롬프트에서 나온다

스냅샷의 토큰 벡터는 항상 히스토리 렌더가 아니라 라이브 프롬프트에서 취한 `prompt_tokens[..boundary]`다. `resolve_history_boundary`가 히스토리 벡터를 라이브 프롬프트 토큰과의 최장 공통 접두사로 잘라내는데, 이 클립이 벡터를 스냅샷이 기술하는 상태의 진짜 접두사로 만들고, 히스토리/스캐폴드 이음새를 가로지르는 BPE 병합으로 불안정해질 토큰도 함께 떨어낸다. 어떤 템플릿 출력이나 메시지 내용도 키가 자기와 안 맞는 상태를 기술하게 만들 수 없다.

### 2.2 렌더러 하나, 스냅샷 생산자 한 쌍

`apply_history_with_kwargs`는 같은 메시지 목록을 `add_generation_prompt`를 끈 채 렌더링하고, 기존 진입점은 같은 내부 렌더에 위임하므로 렌더러는 하나이고 기존 호출자 동작은 그대로다. 스토어 쪽에서는 `donate_finished_sequence_cache`에 인라인돼 있던 캡처와 삽입을 `insert_model_state_snapshot`으로 빼내 경계 캡처와 생성 종료 기증이 공유한다. 기존 기증은 이전과 정확히 같게 동작한다.

### 2.3 비용 계산

`capture_history_boundary_snapshot`은 `prompt_tokens[prefill_start_offset..boundary]` 구간을 한 번 forward하고, 거기서 스냅샷을 떠 삽입한 뒤 `prefill_start_offset`을 전진시킨다. `execute_full_prefill`과 `start_chunked_prefill`은 남은 접미사를 아무 수정 없이 처리한다. forward되는 총 토큰 수는 변하지 않고, 비용은 그래프 런치 한 번과 상태 복사다. 리뷰가 실제 메모리 위험을 하나 제거했다. 세그먼트 forward가 처음에는 `[1, segment_len, vocab]` 로짓 블록 전체를 평가해서 20k 토큰과 150k vocab 기준 수 GB 피크를 만들었는데, 이제 마지막 위치만 평가하고 할당자 캐시를 비운다. 재검증에서 히트 수치가 바이트 단위로 동일했고, 이는 캡처된 상태가 변하지 않았다는 증명이다.

### 2.4 리베이스에서 발견한 교차 이슈 파손

#1151의 supersede 규칙은 같은 세션에서 들어오는 벡터의 진부분 접두사인 저장 스냅샷을 제거한다. 한 턴의 히스토리 경계 벡터는 항상 같은 턴의 완료 벡터(`history + scaffold + sampled reply`)의 진부분 접두사이므로, 모든 경계 스냅샷이 자기 턴의 완료 기증에 의해 삭제됐다. 리베이스 빌드 실측: 둘째 턴이 189 중 0으로 되돌아갔는데 `snapshot_inserts`는 여전히 6이고 `snapshot_evictions_lru`는 0. 생성은 되고 있고 제거가 용량 압력 때문이 아니라는 두 사실을 동시에 증명한다. "같은 세션의 더 긴 벡터가 더 짧은 것을 대체한다"는 전제는 생산자 하나 안에서는 성립하지만 둘 사이에서는 깨진다. 완료 벡터는 더 길지만 거의 쓸 수 없고(꼬리가 바로 이 에픽이 문서화한 발산이다), 경계 벡터는 더 짧지만 항상 쓸 수 있다. 이제 스냅샷이 `SnapshotOrigin`을 지니고 규칙은 한 origin 안에서만 체인을 건다. `ModelSnapshotEntry::new`는 `Completion`을 기본값으로 두어 기존 호출자 동작을 보존한다. 수정 후 같은 프로브가 150/189와 184/214로 복귀했고 상주 엔트리는 4개(각 1개짜리 체인 둘)였다.

### 2.5 게이트와 실패 동작

경계 경로는 `supports_snapshot_reuse()` 모델, 텍스트 전용 요청, 스토어 활성, 그리고 클립된 경계가 `min_prefix_tokens` 이상이면서 프롬프트보다 엄격히 짧을 때만 도달한다. 주 렌더가 `render_simple_fallback`으로 폴백했다면 히스토리 렌더를 버린다(폴백은 템플릿이 아니므로 그 히스토리 형태는 모델이 본 적 없는 프롬프트를 기술한다). 추가 forward의 eval 실패는 기존 prefill eval 실패와 똑같이 시퀀스를 중단시키고, 그 외 모든 거부는 요청을 이전 방식대로 prefill하게 둔다. `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1`은 재빌드 없이 #1143 이전 prefill을 복원하며, 리뷰 수정 후에는 두 번째 렌더와 인코딩도 건너뛰어 `main` 기준선을 셀 단위로 재현한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 완료 벡터 수선이 아니라 재렌더링된 히스토리를 스냅샷

생성 종료 벡터를 수선하는 접근(스캐폴드 제거, 답변 재토크나이즈)은 발산 클래스를 하나씩 쫓아야 하고 템플릿 버릇에 계속 볼모로 잡힌다. 히스토리 렌더 자체의 토크나이즈를 키로 삼으면 다음 턴의 프롬프트가 바로 그 렌더에서 시작하므로 세 클래스가 구성상 무력화된다.

### 3.2 기본 켜짐, 킬 스위치 동반

이 기능이 있어야 해당 계열에서 프롬프트 캐시가 애초에 적중하므로 기본값은 켜짐이다. 비용(해당 요청마다 포그라운드 prefill에 그래프 런치 한 번과 모델 상태 전체 복사)은 단일 턴 트래픽만 받는 배포에는 실질 부담이라, 킬 스위치를 임시 옵트아웃으로 두고 자동 능력 게이트는 #1153으로 분리했다(능력 정보가 워커 스레드 뒤에 있어 HTTP 계층에 공개되지 않는다).

### 3.3 생성 종료 기증은 유지한다

경계 스냅샷이 있으면 완료 기증을 억제하는 안은 검토 후 기각했다. 그 엔트리는 죽은 무게가 아니다. 답변이 정준으로 재토크나이즈된 falcon-h1-tiny에서 더 긴 히트를 만든 것이 바로 그 엔트리다. 두 생산자가 각자의 supersede 체인을 갖고 공존한다.

### 3.4 배치 콜드 코호트는 의도적으로 제외

두 행 이상의 `BatchedCold` 코호트는 행별 분할 지점 없이 모든 행을 한 패스로 forward한다. 그 행들을 non-cold로 표시하면 사실상 이 계열의 모든 채팅 행이 순차 prefill로 빠져 계열 전체의 배치 prefill을 포기하게 되는데, 놓친 재사용보다 비싸다. 해당 메서드에 근거를 기록했다.

### 3.5 리뷰 지적 반영

로짓 실체화와 킬 스위치 범위 수정 외에도: HTTP 쪽 히스토리 토크나이즈 실패가 더는 요청을 영구히 조용히 거부하지 않고(스케줄러의 인코딩이 기회를 가진다), 경계 판정을 순수 함수 둘로 쪼개 테스트를 붙였으며, 이 브랜치가 대화 길이에 비례하게 키운 기존 preemption 누수(`prompt_cache_seq_ctx` 엔트리)를 축출 지점에서 정리했고, 트레이스 스팬이 갓 forward한 경계 토큰을 `cached`로 보고하던 것을 바로잡았으며, 캡처가 컨텍스트 전체를 복제하는 대신 토큰 벡터만 꺼내 간다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 18 |
| 라인 | +1301 / -93 |
| 새 환경변수 | 1 (`MLXCEL_DISABLE_BOUNDARY_SNAPSHOT`) |
| 새 스토어 개념 | `SnapshotOrigin` (생산자별 supersede 체인) |
| prefill당 forward 총 토큰 | 불변 |
| 벤치마크 기록 | `docs/benchmark_results/history-boundary-snapshot-m1ultra-2026-08-14.md` |

### 영역별 변경

**`src/server/chat_template.rs`, `src/server/chat_request.rs`, `src/server/config.rs`**
- 히스토리 렌더(`apply_history_with_kwargs` 계열), 게이트를 갖춘 `PreparedChatRequest::history_prompt`, 기존 prompt/토큰 분리를 그대로 따르는 `PromptCacheRequestContext::{history_prompt, history_prefix_tokens}`.

**`src/server/model_provider.rs`**
- 디스패치 스레드가 같은 `tokenize_prompt_for_generation` 규약으로 히스토리 렌더를 프롬프트 옆에서 토크나이즈. 스케줄러 스레드는 어느 인코딩도 지불하지 않는다.

**`src/server/batch/scheduler.rs`**
- `resolve_history_boundary`, `capture_history_boundary_snapshot`, 기증 경로와 공유하는 `insert_model_state_snapshot`, 전체·청크 prefill이 함께 소비하는 `prefill_start_offset` 전진.

**`src/server/prompt_cache/{entry,policy,store}.rs`**
- 빌더로 지정하는 `SnapshotOrigin`과 origin 범위 supersede, policy의 킬 스위치.

**`src/server/routes/{chat,responses,anthropic}.rs`, `src/server/batch/observability.rs`**
- 여섯 호출부 전부에 연결된 `build_prompt_cache_request_context`, 분할 prefill에서도 `total_prefill_tokens`가 프롬프트 합과 일치하게 하는 `record_prefill_tokens`.

---

## 5. 검증과 후속

### 통과

- `cargo test --release --lib server::prompt_cache`(142), `server::chat_request`(80), `server::chat_template`(100), `server::batch::`(353), `server::routes::`(116), `server::model_provider`(51). `cargo clippy --release --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --check`. 신규 테스트 중 `history_boundary_snapshot_hits_where_the_end_of_generation_one_cannot`이 에픽의 형태를 스토어 수준에서 부호화한다.
- Apple M1 Ultra에서 실물 체크포인트 A/B, 두 팔 모두 프로덕션 `/v1/chat/completions` 경로, 기준선은 `main` 9c154ff3. 귀속은 자기 검증형이다. `snapshot_inserts`가 기준 팔에서는 턴당 1회, 수정 팔에서는 턴당 2회 오른다. qwen3.5-0.8b-4bit: 둘째 턴 `cached_tokens` 0/189에서 150/189, 셋째 턴 0/214에서 184/214, `snapshot_hits` 0에서 2. 히트 길이는 직전 턴의 히스토리 경계와 정확히 일치. dense-KV 대조군 llama-3.2-1b는 전 셀 동일.
- greedy 결정성은 숨기지 않고 특성화했다. 킬 스위치 A/B에서 greedy 한 턴이 168자 동일 후 한 단어 갈라졌고, 각 팔은 세 번 반복에서 바이트 단위로 동일했다. forward 둘은 하나와 같은 순서로 리듀스되지 않는다. 문서화된 근소 동률 뒤집힘 클래스이며 `--prefill-chunk-size`와 캐시 히트에서 이미 현상 유지다.

### 미검증

- 벽시계 지연과 TTFT. 측정 내내 로드 애버리지 22였던 박스라 절대 시간 주장은 성립할 수 없었고, 암시하는 대신 미측정으로 명시했다.
- 31B 규모의 피크 메모리, 동시 대화 예산 압력(#1146의 영역), 그리고 gemma-4-31b-it-4bit. 이 브랜치는 소형 체크포인트만 돌렸으므로 에픽 수준 검증에 넘겼다.
- falcon-h1-tiny는 기록에는 있으나 이 변경의 귀속 증거가 아니다. 답변이 정준으로 재토크나이즈되어 기준 팔도 적중했다. 이미 적중하던 계열을 경계 스냅샷 추가가 깨지 않는다는 사실만 입증한다.

### 후속

- #1153 (이 PR에서 제기): `supports_snapshot_reuse()`가 스레드 경계를 넘어 공개되기 전까지 dense-KV 배포는 버려질 두 번째 렌더와 인코딩을 지불한다.
- 512 MiB 스냅샷 예산 우려는 gemma-4-31b 검증에서 여전히 유효하다. 대화 하나가 이제 정당하게 체인 둘을 보유하므로 #1151만으로 용량 산수가 해소되지 않는다(에픽 실측 단일 31B 스냅샷 307-369 MB).
- 새로 생긴 조용한 거부 경로 둘(히스토리 렌더가 접두사가 아님, 경계가 `min_prefix_tokens` 미만)은 #1147의 분류 리젝트 표면에 속한다. 그 이슈의 진행 중 수정과 충돌하지 않도록 여기서는 넣지 않았다.
