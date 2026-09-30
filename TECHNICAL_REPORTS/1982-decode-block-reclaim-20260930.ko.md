# 기술 보고서: PR (이슈 #1982), 디코드 중 paged 블록 회수

**날짜**: 2026-09-30

**상태**: GB10에서 meta-llama-3.1-8b-instruct-4bit, 4444블록 예산으로 검증 완료, 아직 머지되지 않음

**언어**: Rust

**위험도**: 중간 (paged KV 예산 압박 시 스케줄러 선점 동작이 바뀜)

## 요약

paged 백엔드와 KV 블록 예산(기본값 `--kv-cache-budget auto`)을 쓰는 상태에서, 풀이 가득 찬 채로 디코드 중인 행이 블록 경계를 넘으면 `PagedBlockPool::acquire_block`이 "block budget exhausted"를 반환했다. 이 쓰기는 모델 forward 안에서 일어나고(`write_paged`가 `.expect`를 호출), 그 결과 모델 워커가 panic하여 서버가 모든 요청에 503을 돌려주었다. 이제 디코드는 forward 실행 전에 해당 틱이 새로 만들 블록을 먼저 확보한다. 이때 admission의 회수 루프(콜드 prompt-cache prefix 먼저, 그다음 선점)를 그대로 재사용한다. 실제 서버 테스트에서 같은 예산 때문에 워커가 죽거나 멈추거나 livelock에 빠지는 경로가 다섯 가지 더 드러나(longest-first 선점, 같은 우선순위 admission 간 상호 선점, 헛도는 연기, 예산 관문이 없는 배치 prefill, 디코드에 블록을 빼앗기는 chunked prefill), 이 PR에서 함께 고쳤다.

## 문제 정의

prefill admission(`admit_paged_prefill`)은 블록을 회수했지만, 스케줄러의 다른 경로는 회수하지 않았다. prompt-cache의 paged 엔트리는 예산에 포함되는 블록을 고정하고 있고, #1978이 기본 저장 용량을 키웠기 때문에 디코드 증가가 한도에 닿는 일이 더 잦아졌다.

## 변경 요약

- `mlxcel-core`: `PagedBlockPool::blocks_to_append`와 `CachePool::paged_blocks_to_append`가 append 시 레이어별로 새로 얻을 블록 수를 센다. 새 tail 블록 수에, 첫 새 토큰이 공유된 부분 tail 블록에 쓰일 경우의 copy-on-write 분기 1개를 더한다.
- 새 파일 `scheduler/block_reclaim.rs`: 회수 루프를 `admission.rs`에서 꺼내 `reclaim_blocks`로 옮겼다. 작은 `PagedBlockReclaimer` 트레이트 위에 작성해서, 실제 `CachePool`과 `PromptCacheStore`로 테스트할 수 있다. admission과 decode 모두 이 함수를 호출하며, 루프는 여전히 하나뿐이다.
- `execute_decode_step`은 먼저 `reserve_decode_step_blocks`를 호출한다. 풀에 여유가 있으면 부작용 없는 읽기 전용 확인으로 끝난다. 압박 상태에서는 미리 만들어 둔 lookahead를 폐기하고, 선점 하한 1로 회수한다(마지막 행은 자기 자리를 만들려고 선점되지 않는다). 선점으로 빠진 행은 제외하고, 그래도 늘어날 수 없는 행은 연기하거나 제외(shed)한다. 제외된 행은 "KV cache budget exhausted" 오류로 끝나므로 forward가 소진된 풀에 대해 실행되지 않는다. lookahead prime은 추가 위치에 블록이 필요하면 건너뛴다.
- 블록 회수용 희생자 순서(`select_block_reclaim_victim_from`): 우선순위가 낮은 행, 그다음 생성 토큰 수가 가장 적은 행, 그다음 가장 새 id다. 설정된 슬롯 선점 정책과 의도적으로 다르게 했다. `LongestFirst`는 완료에 가장 가까운 행을 선점하는데, 첫 GB10 실행에서 매번 그렇게 되었다. 10분 동안 선점 179회에 완료는 1회였고, 희생자는 모두 약 250토큰을 생성한 상태였다. 슬롯 선점은 설정된 정책을 그대로 쓴다.
- admission은 이제 우선순위가 엄격히 낮은 행만 선점한다. 이전에는 동시에 들어갈 수 없는 같은 우선순위 요청 두 개가 매 prefill 직후 서로를 선점했다(생성 토큰 1개인 행의 선점이 1,600회 이상). admission이 헤드 요청을 연기하면 같은 틱에 디코드 스텝을 실행한다. 이 스텝이 없을 때는 `decide_action`이 계속 Prefill을 골라 연기만 반복했고 블록이 한 번도 해제되지 않았다(CPU 98% 정체).
- 배치 prefill에는 예산 관문이 없었다. 1095토큰 프롬프트 세 개가 함께 admission되어 워커가 panic했다. 이제 예산에 맞지 않는 헤드는 단일 시퀀스 admission 경로로 보내고, 윈도우 drain은 예산을 넘길 첫 행에서 멈춘다.
- 진행 중인 chunked prefill의 남은 블록을 따로 떼어 둔다(`available_paged_blocks`). 그 admission은 프롬프트 전체를 기준으로 확인했지만 블록은 청크 단위로 얻기 때문이다. 이전에는 디코드 증가가 이 블록을 소모해서 다음 청크가 panic했다. 이 예약분 때문에만 블록이 부족한 행은 제외하지 않고 한 틱 연기한다. 해당 prefill은 배치에 합류하면 선점 대상이 된다.
- `src/models/sanitize_tests.rs`: 테스트가 이제 조건 없이 호출하는 헬퍼에 붙어 있던 `cfg(not(feature = "cuda"))` 게이트를 제거했다. 이 수정이 없으면 현재 main에서 `cargo test --features cuda --lib`가 컴파일되지 않는다.

## 검증

- 단위 테스트: `block_reclaim_tests.rs`에 11개 케이스가 있다. 캐시된 prefix로 가득 찬 풀에서 디코드 행이 블록 경계를 넘는 경우, 회수 없이는 append가 실패하고("block budget exhausted") 회수가 있으면 LRU 엔트리 정확히 하나를 제거한 뒤 성공한다. 예산에 여유가 있으면 아무것도 제거하거나 선점하지 않는다. 제거할 엔트리가 없으면 선점한다. 회수할 것이 전혀 없으면 행을 제외하며 panic은 없다. 나머지 케이스는 블록 중간 위치의 행이 제외되지 않는 경우, admission 우선순위 제한, chunked prefill 예약(디코드가 예약된 블록을 가져가지 않고 제거를 선택), 연기, 배치 윈도우의 예산 차감을 다룬다. `paged_append_need_tests.rs`에는 copy-on-write 분기를 포함해 3개, `scheduler_tests.rs`에는 선택자 케이스 3개가 있다.
- `cargo test --release --features cuda --lib -- server::batch server::prompt_cache models::sanitize_tests --test-threads=1`: 674개 통과. mlxcel-core `cache::paged`: 132개 통과. `cargo clippy --release --features cuda --lib --tests -- -D warnings`와 `cargo fmt --all` 모두 문제없다.
- GB10, `--kv-cache-budget 600000000 --decode-storage-backend paged --max-batch-size 4`(4444블록)에서 prefix 하나를 먼저 캐시에 올려 둔 뒤, 600토큰을 생성하는 3턴 대화 3개를 동시에 실행했다. 9턴 모두 오류나 panic 없이 완료되었고, 디코드 회수 패스 16회와 선점 12회가 있었다. 이 브랜치의 이전 단계에서는 같은 부하가 위에 나열한 실패 모드를 각각 일으켰다.
- 같은 서버에서 캐시된 prefix가 남아 있는 상태로 1099토큰 프롬프트에 2200토큰을 생성했다. 로그에는 디코드 회수 패스 세 번이 기록되었고 각각 `evicted_prefixes=1`, `preempted=0`이었으며, 요청은 완료되었다.
- 기본 실행(auto 예산, 753620블록)에서 3턴 대화 2개를 실행했다. 2턴과 3턴이 prompt cache를 적중했고(캐시 토큰 1376개와 1696개), 회수나 선점 로그는 하나도 없었다.

## 남은 작업

- `free_block_budget`은 풀 전체 개수를 세지만, 해제된 행은 레이어별 free list에 들어간다. 균일한 paged 레이아웃에서는 리스트가 균형을 유지하므로 전역 개수가 실제로 맞다. 균일하지 않은 레이아웃이라면 특정 레이어에서 여전히 실패할 수 있다.
- MixedStep 틱의 prefill 청크는 틱 단위 예약이 아니라 chunked prefill 예약으로 보호된다. speculative slice 라운드는 모델 소유 캐시를 쓰므로 paged 풀 범위 밖이다.
- 이 정도로 빡빡한 예산에서는 선점이 재계산 비용을 낳는다. 압박 실행에서 9턴에 선점 12회였다. 디코드용 여유를 남기는 admission 워터마크를 두면 줄일 수 있지만, 이 PR에서는 구현하지 않았다.
