# PR #2097: 풀 기반 `KVCache::trim`의 블록 테이블 되감기

**작성일**: 2026-10-06
**상태**: 리뷰 완료, CUDA(GB10)에서 검증; M5 타일 정렬 경로는 기여자 검증만 있음
**위험도**: 낮음

## 요약

`KVCache::trim`은 풀 기반 캐시(`paged_backing.is_some()`)에 대해 `0`을 반환했습니다. 그래서 `trim(excess)`로 prefill 패딩 행을 제거하던 모든 호출부가 패딩 행을 풀에 남겼고 `offset`도 패딩된 길이에 머물렀습니다. 그 결과 decode가 패딩 행까지 attention 대상으로 삼았고, RoPE 위치도 패딩된 `offset`에서 가져왔습니다. 이 PR은 `trim`이 캐시 자신의 backing을 통해 `PagedBlockPool::rewind_tokens`를 호출하고 실제로 제거된 개수만큼 `offset`을 줄이도록 바꿉니다. 또한 decode lookahead 해제(`apply_lookahead_trim`)가 풀 전용 `rewind_paged_tokens` 분기 대신 같은 호출을 쓰도록 합니다. 기존 분기는 풀만 되감고 `offset`은 한 칸 앞에 남겨 두었습니다.

Closes #2096. 기여자: rapsealk.

## 리뷰

- 풀 기반 캐시에 대한 모든 실제 `trim` 호출부(`scheduler/prefill.rs`의 패딩 trim 네 곳과 lookahead 해제)는 실제 trim을 기대합니다. `trim`과 별도의 풀 되감기를 함께 호출하는 곳이 없으므로 이중 되감기는 생기지 않습니다. `trim_to`/`DetachedCacheSet::truncate_to`에는 실제 호출부가 없습니다.
- `sync_paged_state_with_dense`는 풀 기반 시퀀스에서 바로 반환하므로, 해제 뒤의 `sync_sequence_storage` 호출이 블록 테이블을 다시 자르지 않습니다.
- `live_len()` 제한은 풀 분기보다 먼저 적용되고, `offset`은 풀이 실제로 제거한 개수만큼 움직이므로 이 경로로 풀 `len`과 `offset`이 어긋날 수 없습니다.
- 풀 분기는 `.expect`를 쓰지만 기존 lookahead 분기는 경고 로그만 남겼습니다. 같은 풀을 쓰는 인접한 `write_paged`, `update_and_fetch_paged` 호출과 같은 방식이며, 여기서 되감기가 실패한다면 블록 테이블이 이미 불일치 상태입니다. MEDIUM으로 기록만 하고 수정하지 않았습니다.
- `CachePool::rewind_paged_tokens`는 이제 테스트 외 호출부가 없지만 그대로 두었습니다.

CRITICAL/HIGH 문제는 없었고, 기여자 커밋 위에 코드 변경을 추가하지 않았습니다.

## GB10 검증 (CUDA, release 프로필, `--features cuda`)

브랜치 헤드를 `origin/main`의 `2bbf192f`와 병합한 상태에서 검증했습니다.

- 새 테스트 `trim_drops_padded_prefill_rows_from_a_pool_backed_cache`: 수정 적용 시 통과합니다. `cache.rs`와 `decode_tick.rs`를 `origin/main`으로 되돌리면 첫 단언에서 실패합니다(`left: 0, right: 27`).
- 인접 mlxcel-core 테스트, `gpu-lock` 아래 `--test-threads=1`: `cache::paged_batch_decode` 23개, `cache::paged` 137개, `cache::tests` 83개, `cache::detach` 51개, `speculative` 175개 모두 통과. 메인 크레이트: `lookahead` 10개(해제 위치 테스트 포함), `block_reclaim` 23개, `server::batch::scheduler` 167개(7개 ignored) 통과.
- `cargo fmt --all -- --check`, `cargo clippy -p mlxcel-core --lib --tests --features cuda -- -D warnings`, `-p mlxcel` 동일 명령 모두 경고 없음.
- 서버: `mlxcel-server -m llama-3.2-1b-4bit --no-prompt-cache`로 서로 다른 채팅 프롬프트 5개(프롬프트 토큰 40, 42, 51, 52, 64)를 동시에 보내는 라운드를 3회 반복했습니다(`temperature 0`, `max_tokens 32`). 기본 백엔드(auto, 이 호스트에서는 풀 기반 paged로 결정)와 `--decode-storage-backend dense`를 비교했습니다. 디버그 로그로 패딩된 batched prefill이 실제로 실행됐음을 확인했습니다(`batched prefill: 3 requests, padded to 52` 등).

| 빌드 | 기본 vs dense, 동일한 출력 | 첫 batched decode gather |
|---|---|---|
| `origin/main` (수정 되돌림) | 15개 중 5개 | 요청 4개에 걸쳐 KV 토큰 224개 (패딩 행 잔존) |
| PR #2097 | 15개 중 15개 | 요청 4개에 걸쳐 KV 토큰 189개 (실제 행과 요청당 1개) |

수정 전 기본 백엔드는 `Repeat: one two three`에 `One!Two!Three!`로, 프랑스 질문에는 `The capital of France is... Paris! The City of Light, ...`처럼 장황하게 답했으며, 이는 기여자의 M5 보고와 일치합니다. dense 기준 출력은 두 빌드에서 바이트 단위로 같았습니다. 이로써 Neural Accelerator가 없는 하드웨어에서도 batched prefill의 최장 행 패딩이 풀 기반 출력을 오염시킨다는 이슈의 주장을 확인했습니다.

## 이 호스트에서 검증하지 못한 항목

- M5 타일 정렬 prefill 경로(`should_align_prefill()`가 true)는 Apple M5 하드웨어가 필요합니다. 기여자의 M5 Pro 결과(Llama-3.2-1B, SmolLM2-135M, Qwen3-0.6B, 수정 전 5개 중 0-1개, 수정 후 5개 중 5개)는 보고된 대로 받아들입니다.
- M5에서의 #1760 전체 프롬프트 재생 abort는 이 PR의 범위가 아닙니다.
