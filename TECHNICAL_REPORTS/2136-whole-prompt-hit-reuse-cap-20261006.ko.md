# PR #2136: 프롬프트 전체 캐시 히트가 마지막 프롬프트 토큰을 중복시키지 않도록 수정

**작성일**: 2026-10-06
**상태**: CUDA(GB10)에서 검증 완료. M5 tile-padded abort 경로는 이 호스트에서 재현 불가
**위험도**: 낮음

## 요약

클라이언트가 동일한 프롬프트를 다시 보내면 prompt cache가 프롬프트의 모든 토큰과 일치한다. `try_adopt_cached_prefix`는 `len`개 토큰을 모두 복원했고, admission은 sampler가 logit을 얻도록 prefill 커서만 `len - 1`로 되돌렸다. 그 결과 prefill은 이미 캐시에 들어 있는 마지막 프롬프트 토큰을 `len - 1`이 아닌 `len` 위치에 한 번 더 넣었다. 이 PR은 snapshot 경로와 KV 경로 모두에서 adopt 길이를 `len - 1`로 제한해, 다시 실행되는 토큰이 제자리에 들어가게 한다.

Closes #1760.

## 1. 문제 정의

이슈 #1760은 두 가설을 열어 두었다. (a) 캐시에 실제 중복이 있다. (b) warm 요청은 토큰 1개만, cold 요청은 프롬프트 전체를 forward하는 데서 오는 forward 폭 반올림 효과일 뿐이다. admission clamp 직후 측정한 layer별 캐시 offset으로 (a)임을 확인했다.

| 체크포인트 | 경로 | prompt_len | clamp 직후 offset |
|---|---|---|---|
| gemma-3-4b-it-4bit | snapshot (truncating restore) | 115 | 115 |
| gemma-4-12b-it-4bit | snapshot (truncating restore) | 119 | 119 |
| qwen3-4b-4bit | paged KV | 128 | 128 |
| llama-3.2-1b-4bit | paged KV | 160 | 160 |

KV 경로는 clone 경로가 블록 단위로 내림하기 때문에, 프롬프트 길이가 paged 블록(기본 32)의 배수일 때만 프롬프트 전체 히트가 된다. #2096이 M5에서 32/48 토큰 replay에서만 abort를 본 것과 맞는다. 그 abort는 padded mask를 `len - 1`개의 캐시 행 기준으로 만들었는데 캐시에는 `len`개가 있어서 발생했다.

## 2. 변경 내용

- `src/server/batch/scheduler/prompt_cache.rs`에서 store lookup 직후 `whole_prompt_reuse_cap(matched_len, prompt_len) = min(matched_len, prompt_len - 1)`을 적용한다.
- Snapshot 경로: 상한 때문에 일치 길이가 줄어들면 모델에 `snapshot_truncatable_to(snapshot, len - 1)`을 묻는다. 모델이 동의하면 `restore_sequence_state_truncated`로 복원한다. 거부하면(recurrent state, 또는 wrap된 sliding ring) cold prefill로 되돌아가고 저장된 엔트리 길이와 함께 `layout_constraints` reject를 기록한다. 부분 복원은 설치하지 않는다.
- KV 경로: 상한을 dense `truncate_to`와 paged 블록 내림보다 먼저 적용하므로, 이후 단계는 `len - 1`개 이하만 설치한다. 상한이 `min_prefix_tokens`보다 작으면 `prefix_too_short`로 거절한다.
- 멀티모달 whole-entry 게이트(#124 step c)는 상한 적용 값이 아니라 store가 보고한 일치 길이를 기준으로 판단한다. 그래서 whole-entry VLM replay는 계속 adopt되며, 남는 토큰 1개짜리 suffix는 이전 back-off가 이미 token 경로로 forward하던 것과 같다.
- `admission.rs`의 back-off는 안전장치로 남기고 warn으로 로그한다. 이 경로에 도달하면 불변식이 깨졌다는 뜻이다.
- 부수 효과: 프롬프트 전체 히트에서 `usage.prompt_tokens_details.cached_tokens`와 shared budget 계산이 실제로 복원한 길이, 즉 이전보다 1 작은 값을 보고한다.

## 3. 기술적 선택: admission이 아니라 adopt에서 제한

이슈가 제안한 방향과 같다. 복원이 이미 토큰을 설치한 뒤라서 prefill 커서만 옮겨서는 해결되지 않는다. `len - 1`까지만 복원하면 모든 family에서 forward가 올바른 슬롯에 들어가며, 그것이 가능한지는 store가 이미 쓰는 truncation predicate가 판단한다. 거부 사유는 기존 `layout_constraints`로 기록해, 거절 지점 하나 때문에 고정된 reject enum과 `/v1/cache/stats` 스키마를 넓히지 않았다.

## 4. 검증 (GB10, CUDA, `mlxcel-server`, prompt cache on, temperature 0, seed 0, 200 토큰)

| 체크포인트 | 수정 전 | 수정 후 |
|---|---|---|
| gemma-3-4b-it-4bit | 226번째 문자에서 분기 | 3/3 cold와 바이트 단위 동일 |
| llama-3.2-1b-4bit | 230번째 문자에서 분기 | 3/3 바이트 단위 동일 |
| gemma-4-12b-it-4bit | 71번째 토큰에서 분기 | 27번째 토큰에서 분기, bf16 tie (warm top-2 gap 0.0) |
| qwen3-4b-4bit | 76번째 토큰에서 분기 | 82번째 토큰에서 분기, bf16 tie (warm top-2 gap 0.0) |
| qwen3.5-0.8b-4bit (recurrent) | 동일 (boundary 엔트리 히트) | 동일 |

Gemma 4와 Qwen3에 남은 분기는 forward 폭 효과 (b)다. warm 요청은 토큰 1개(Gemma 4) 또는 32개(Qwen3, paged 블록 내림 이후)를 forward하고 cold는 더 넓은 chunk를 forward한다. 폭 차이를 없앤 대조 실험(양쪽 모두 `--prefill-chunk-size 1`, Gemma 4는 boundary snapshot 비활성화)으로 두 원인을 분리했다. 수정 후에는 두 체크포인트 모두 토큰이 동일하고 logprob도 비트 단위로 같다. 수정 전에는 둘 다 여전히 분기하며, Gemma 4는 첫 토큰의 logprob부터 다르다(-0.5 vs -0.625).

테스트:

- `src/server/batch/scheduler_whole_prompt_hit_tests.rs`: tiny 실제 Gemma 3로 세 테스트를 돌린다. 모델 자체 snapshot에서 복원된 offset을 읽어 동일 replay(truncating), admission을 거친 whole-entry 히트(`prefill_start_offset == already_cached_tokens == len - 1`), sliding window 거부 시 cold fallback과 reject 집계를 확인한다.
- `tests/prompt_cache_e2e.rs::identical_prompt_replay_restores_all_but_the_last_token`: ignored 테스트로, qwen3-0.6b-4bit와 `--apc-block-size 1`로 실행한다. `cached_tokens == prompt_tokens - 1`과 바이트 단위로 동일한 응답을 확인하며, 수정 전 바이너리에서는 실패한다(20 vs 19).

## 5. 이 호스트에서 검증하지 못한 항목

- `trinity-nano-preview-4bit`(AFMoE)가 로컬에 없다. 대신 Qwen3와 Llama 3.2로 KV 경로를 검증했다.
- M5 broadcast abort는 Metal 하드웨어가 필요하다.
- 실제 체크포인트에서 거부 경로는 chat 요청으로 도달하지 못했다. recurrent family가 더 짧은 boundary 엔트리에서 히트하기 때문이며, 이 경로는 단위 테스트가 다룬다.

## 6. 관련 항목

- #1754: completion-origin snapshot의 token 수와 캐시가 실제로 가진 offset의 차이. 위 측정에서도 Gemma 4와 KV 경로의 `length` 종료는 `offset == token_len - 1`이었다.
- #2096 / PR #2097: M5 abort를 보고한 pool-backed trim 수정.
