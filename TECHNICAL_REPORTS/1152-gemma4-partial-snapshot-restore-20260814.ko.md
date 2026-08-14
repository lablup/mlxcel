# 기술 보고서: PR #1152 - feat(models): restore Gemma 4 snapshots at the longest common prefix

**날짜**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: 낮음 (계열별 옵트인. 능력 질의의 기본 답이 `false`라서 순환 상태 계열과 dense 계열은 구성상 exact-prefix 의미를 유지)

---

## 요약

스냅샷 경로는 저장 엔트리가 들어오는 요청의 exact prefix일 것을 요구했기 때문에, 긴 공통 접두사를 공유하면서 엔트리 끝 전에 갈라지는 엔트리는 쓸모가 없었다. 회전 어텐션 계열에는 이 요구가 하드웨어가 실제로 요구하는 것보다 엄격하다. "잘라낼 수 없다"는 제약은 슬라이딩 레이어의 링 버퍼가 wrap된 뒤에만 성립한다. wrap 전의 링은 선형이라 잘라내기가 dense와 기계적으로 동일하며, 이것이 바로 업스트림 mlx-lm의 `can_trim_prompt_cache`가 검사하는 조건이다.

이번 변경은 Gemma 4에서 exact-prefix 매칭을 최장 공통 접두사 매칭으로 바꾸되, 순환 상태 계열은 전혀 건드리지 않는다. 능력은 `LanguageModel::snapshot_truncatable_to`로 표현되고 기본값이 `false`라 계열이 명시적으로 옵트인해야 한다. Gemma 4는 스냅샷 자체의 스칼라에서 레이어별로 답한다. 스토어는 `min_prefix_tokens`를 넘는 최선의 발산 후보를 최장 공통 접두사에서 채택하고, 거절은 #1147의 분류된 `snapshot_diverged` 리젝트로 남는다. gemma-4-e2b-it-4bit 실물 검증이 합성 픽스처로는 도달할 수 없는 진짜 버그를 잡았다. 아무것도 저장하지 않는 KV 공유 레이어가 오프셋 0을 상대로 잘라내기 복원에 실패했다.

---

## 1. 문제 정의

### 1.1 배경

`PromptCacheStore::lookup_snapshot_prefix`는 저장 토큰 벡터가 요청의 exact prefix여야 한다고 요구했다. 에픽 #1148은 snapshot-only 계열에서 저장 벡터가 다음 턴과 긴 접두사를 공유하다가 템플릿 산출물이나 재토크나이즈 드리프트에서 갈라지는 일이 일상임을 보였고, 그때마다 엔트리 전체가 버려졌다. Gemma 4는 60개 레이어를 5:1 슬라이딩:풀 패턴으로 돌리고 31B 체크포인트의 `sliding_window`는 1024라서, 짧고 중간 길이의 대화에서는 모든 레이어가 아직 선형이고 발산 지점까지 기계적으로 잘라낼 수 있다. 업스트림 mlx-lm이 이미 이를 부호화했다. `can_trim_prompt_cache`는 회전 캐시가 unwrapped일 때만 trim을 허용한다.

### 1.2 기존 문제

- **긴 공통 접두사를 통째로 버렸다.** 저장 139토큰 중 90을 공유하는 후보가 아무 기여도 못 하고 요청이 0부터 다시 prefill했다.
- **제약을 잘못된 granularity로 적용했다.** "회전 캐시는 잘라낼 수 없다"는 wrap 후에만, 레이어별로만 참이다. 이를 계열 전체의 절대 규칙으로 다루면서 모든 짧은 대화 케이스를 포기했다.
- **재토크나이즈 드리프트를 구제할 수 없었다.** 거의 일치하는 접두사 끝부분의 작은 발산(falcon-h1이 보인 클래스: 샘플링 120 대 재토크나이즈 118 토큰)이 엔트리 전체를 잃게 했다.
- **순환 상태 계열을 끌어들이면 안 된다.** GatedDeltaNet과 SSM 상태는 임의 경계로 잘라내는 것 자체가 불가능하다. 스토어의 휴리스틱으로 능력을 추론하는 설계는 이들을 오염시킬 위험이 있었다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|---|---|---|
| wrap됐거나 비선형인 캐시를 잘라내 생성을 조용히 오염 | 높음 | 낮음 (독립적인 두 실패 모드를 각각 배제하고 테스트) |
| 순환 상태 계열이 잘라내기 복원을 시도 | 높음 | 낮음 (기본 능력이 `false`, 계열별 옵트인) |
| 채택된 접두사의 상태가 같은 접두사의 콜드 prefill과 다름 | 높음 | 낮음 (출력 동일성 테스트가 정합성 기준) |
| dense-KV 계열에 영향 | 낮음 | 낮음 (스냅샷 경로에 진입하지 않음을 라이브로 확인) |

---

## 2. 기술 검토

### 2.1 선형성의 실패 모드는 둘이고, 둘 다 배제했다

`RotatingKVCacheSnapshotState::is_unwrapped` / `can_truncate_to`와 라이브 캐시에서 업스트림을 반영하는 `RotatingKVCache::is_trimmable`은 독립적인 두 조건을 배제한다. 하나는 `idx`가 더는 `offset`을 따라가지 않는 wrap된 링이고, 다른 하나는 더 미묘한 함정으로, 윈도우 초과 prefill 뒤 한 스텝 동안 `update_concat`이 남겨 두는 과대 임시 버퍼다. 후자는 `idx == offset`을 유지해서 순진한 unwrapped 검사를 통과하지만, 다음 in-place 업데이트가 뒤에서부터 다시 슬라이스하므로 잘라낸 뒤 데이터가 어긋난다. 유닛 스위트가 라이브 캐시를 wrap 지점까지 몰아가며 두 거절을 모두 고정한다.

### 2.2 능력은 스토어 휴리스틱이 아니라 모델의 답이다

`LanguageModel::snapshot_truncatable_to`와 `restore_sequence_state_truncated`의 기본값은 `false` / `Err`다. 아무 말도 하지 않는 계열은 exact-prefix 의미를 유지한다. Gemma 4는 스냅샷 자체의 스칼라에서 레이어별로 답하고, `restore_sequence_state_truncated`는 호출자를 신뢰하는 대신 다시 검사한다. 스토어는 클로저로 능력을 묻되 저장 길이가 아니라 채택 길이에 대해 묻는데, 전용 테스트가 이를 고정한다. 이슈의 요구를 구조로 만든 것이다. 미래의 하이브리드 계열은 명시적으로 옵트인하거나 아예 참여하지 않는다.

### 2.3 스토어 변경은 작고 순서가 있다

`lookup_snapshot_outcome`은 `min_prefix_tokens`를 넘는 최선의 발산 후보를 최장 공통 접두사에서 채택한다. exact-prefix 후보는 공통 접두사가 더 긴 발산 후보보다 항상 우선하고, 능력 거절은 정확히 #1147의 분류 리젝트로 강등된다. 그래서 진단 표면이 정직하게 남는다. "갈라졌고 채택 불가"가 여전히 보이되, 이제 그 이유는 메커니즘 부재가 아니라 wrap된 캐시다.

### 2.4 실물 체크포인트만 찾을 수 있던 버그

첫 gemma-4-e2b 실행이 `layer15: Gemma 4 truncate: target 34 exceeds cached offset 0`으로 실패했다. Gemma 4의 KV 공유 레이어는 Q만 계산하고 K/V를 앞 레이어에서 빌리므로 `snapshot_into`가 건너뛰고 `restore_from`이 빈 상태로 둔다. 잘라내기 가드 `target_len <= offset`이 실제로는 건전한 복원에서 오프셋 0을 상대로 실패한 것이다. 단일 레이어 합성 픽스처는 설계상 아무것도 저장하지 않는 레이어를 표현할 수 없다. 수정은 빈 레이어를 no-op으로 만드는 `Cache::is_populated` 가드이고, 회귀 테스트가 부정 케이스까지 덮어 no-op이 전면 우회로 번지지 못하게 한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 정책을 발명하는 대신 mlx-lm 선례를 따른다

업스트림의 `can_trim_prompt_cache`는 같은 물리적 사실, 즉 unwrapped 링은 선형이라는 점을 담고 있다. 이를 반영하면 Rust 구현을 레퍼런스와 diff하기 쉽고, 이미 프로덕션에서 검증된 조건을 들여온다.

### 3.2 대화 길이 휴리스틱이 아니라 타깃 지점의 레이어별 검사

"1024토큰 미만" 같은 전역 규칙은 너무 엄격하고(풀 어텐션 레이어는 언제나 잘라낼 수 있다) 깨지기 쉽다(윈도우가 체크포인트마다 다르고 e2b는 512다). 검사는 잘라내기 타깃에서 각 슬라이딩 레이어의 스칼라를 읽으며, 5:1 레이어 패턴과 체크포인트 차이를 가로질러 올바름을 유지하는 유일한 정식화다.

### 3.3 exact-prefix 후보가 우선권을 지킨다

부분 채택은 구제책이지 선호 경로가 아니다. exact-prefix 엔트리와 공통 접두사가 더 긴 발산 엔트리가 공존하면 exact 쪽이 이기고, 평범한 복원으로 충분할 때 잘라내기 복원과 그 추가 검사를 피한다.

### 3.4 정합성 기준은 카운터가 아니라 출력 동일성

`gemma4_truncated_restore_matches_a_cold_prefill_of_the_same_prefix`는 N으로 잘라낸 복원이 그 N토큰만 prefill한 상태와 동일하게 디코드할 것을 요구한다. 카운터는 경로가 돌았음을 증명하고, 동일성은 올바르게 돌았음을 증명한다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 10 |
| 라인 | +927 / -33 |
| 새 `LanguageModel` 메서드 | 2 (`snapshot_truncatable_to`, `restore_sequence_state_truncated`, 둘 다 안전한 기본값) |
| 옵트인 계열 | 1 (Gemma 4, 두 VLM 래퍼 포함) |
| 순환 상태 계열의 동작 변화 | 0 (음성 대조로 검증) |

### 영역별 변경

**`src/lib/mlxcel-core/src/cache.rs`**
- `RotatingKVCacheSnapshotState::{is_unwrapped, can_truncate_to}`, 업스트림을 반영한 `RotatingKVCache::is_trimmable`, 윈도우 초과 `update_concat` 함정 배제.

**`src/lib/mlxcel-core/src/generate.rs`**
- `false` / `Err`를 기본값으로 하는 `LanguageModel::snapshot_truncatable_to` / `restore_sequence_state_truncated`.

**`src/models/gemma4.rs`**
- 스냅샷 스칼라 기반 레이어별 능력, `Cache::truncate_to`, KV 공유 레이어를 위한 `Cache::is_populated`, 재검사하는 잘라내기 복원.

**`src/server/prompt_cache/store.rs`, `src/server/batch/scheduler.rs`**
- 클로저로 능력을 받아 최장 공통 접두사에서 채택하는 `lookup_snapshot_outcome`, 저장 길이보다 짧은 `matched_len`을 잘라내기 복원으로 보내는 스케줄러.

**`src/loaded_model.rs`, `src/vision/gemma4_unified.rs`, `src/vision/gemma4_vl.rs`**
- 모든 래퍼를 관통하는 위임.

---

## 5. 검증과 후속

### 통과

- `mlxcel-core::cache::rotating_truncation_tests`: 9 통과. `offset == max_size`의 포함 경계(수용) 대 한 토큰 뒤(거절), 라이브 윈도우 안 타깃이라도 wrap된 캐시는 거절, 윈도우 초과 prefill 함정, 버퍼드 스펙큘러티브 모드, 비 FP16 저장, wrap 지점까지 몰아간 라이브 캐시.
- `models::gemma4_tests::snapshot_prompt_cache`: 7 통과. 콜드 prefill 동일성 테스트와 KV 공유 빈 레이어 회귀 포함.
- `server::prompt_cache`: 165 통과. 부분 채택, exact-prefix 우선권, `min_prefix_tokens` 하한, 옵트인하지 않은 모델의 exact-prefix 유지, 채택 길이 기준 능력 질의.
- `cargo clippy --lib --tests -- -D warnings`, `cargo fmt --check`.
- `/v1/chat/completions` 경로의 실물 체크포인트, `/v1/cache/stats` 카운터 검증. gemma-4-e2b-it-4bit 둘째 턴이 77에서 채택(`matched=77 stored=129 partial=true`), 의도적으로 발산시킨 턴이 66에서 채택(`matched=66 stored=168 partial=true`). 둘 다 저장 엔트리보다 엄격히 짧은 채택으로, 이전 코드에서는 불가능한 동작이다. qwen3.5-0.8b-4bit가 핵심 음성 대조다. `partial=true`가 0회, `snapshot_hits` 0 고정, 대신 `snapshot_diverged` 리젝트 발화. llama-3.2-1b는 내내 `snapshot_lookups = 0`이라 dense-KV 계열은 이 코드를 아예 만지지 않는다.

### 미검증

- 이슈 수용 기준이 지목한 gemma-4-31b-it-4bit 시나리오는 같은 회전 어텐션 계열이며 512토큰 윈도우로 동일 코드 경로를 지나는 gemma-4-e2b-it-4bit로 대신 검증했다. 31B 실행은 오케스트레이터의 대형 모델 패스 몫이다.
- 부분 채택의 벽시계 이득은 재지 않았다. 이 변경은 정합성과 카운터로 검증했다.

### 후속

- wrap 거절은 현재 `snapshot_diverged` 분류를 재사용한다. 운영자가 "갈라졌고 wrap이라 채택 불가"와 "갈라졌고 능력 없음"을 구분해야 한다면 #1147 표면에 하위 사유를 추가할 수 있다.
- 미래의 회전 또는 하이브리드 계열은 `LanguageModel` 메서드 둘을 구현하면 부분 복원을 얻는다. 기본값이 구현하지 않은 계열의 불변을 보증한다.
