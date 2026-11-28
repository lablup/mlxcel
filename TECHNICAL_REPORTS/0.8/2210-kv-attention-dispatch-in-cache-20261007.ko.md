# 기술 보고서: PR #2210 - KV attention 분기를 캐시 안으로

**작성일**: 2026-10-07

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Rust, Markdown(ADR, 문서)

**위험도**: 중간. Qwen3, Llama 3, Helium, DeepSeek V2의 attention 단계가 모두 캐시의 진입점 하나를 거친다. FP16 dense 경로와 pool 기반 경로는 이전과 같은 연산을 수행하고, 실제 체크포인트 패리티도 그대로다. 배치 경로 두 곳(FP16이 아닌 paged 레이아웃에서 FP16로 남는 경계 층, Turbo 행)은 같은 attention을 다른 연산 순서로 계산한다.

## 요약

이 PR 전에는 모델 forward가 직접 캐시의 저장소 종류(`is_paged_backed()`)를 묻고 스케줄러의 `DecodeBatchContext`를 읽어 attention 커널을 골랐다. epic #2166은 저장소가 모델마다 갈라지는 분기가 아니라 시퀀스의 속성인 엔진 하나를 목표로 한다. Phase 4a(#2171)인 이 PR은 `KVCache::attend`와 `cache::attend_batched`를 추가한다. 두 함수는 그 단계의 K/V를 붙이고 캐시 뒤의 저장소를 보고 커널을 고른다. MLA latent 캐시에는 `MlaLatentCache::attend`를 추가했다. Qwen3, Llama 3(그리고 그 attention을 재사용하는 계열), Helium, DeepSeek V2의 absorbed decode가 이 진입점을 호출한다. 설계는 ADR 0008에 남겼고, Gemma 3, Llama 4, `model_owned` 헬퍼는 Phase 4b로 미뤘다.

## 1. 문제 정의

- 커널 선택이 모델마다 `forward_split_attention`과 배치 forward에 반복되어 있었다. pool 기반 캐시는 pooled paged 커널로, paged 레이아웃 아래의 dense 캐시는 dense 포인터 paged 경로(`paged_decode_attention_dense_compat`)로, Turbo 모드는 dequant-first 변형으로, 나머지는 SDPA로 갔다.
- 호출 지점마다 경로를 따로 고르니 같은 모델, 같은 요청에서도 CLI와 서버가 다른 커널을 돌렸다.
- 저장소를 추가하거나 바꾸려면 그 저장소로 분기하는 모델 계열을 모두 고쳐야 했다.

## 2. 변경 요약

- **`src/lib/mlxcel-core/src/cache/attend.rs`**
  - `KVCache::attend(q, k, v, scale, mask)`. pool 기반 캐시에서 mask 없는 한 토큰이면 #899 pooled 진입점을 쓰고, 이 진입점은 쓰기 전에 거절 여부를 정한다. pool 기반의 나머지 경우는 pool intercept와 SDPA. dense Turbo는 dequant-first 변형(ADR 0002). dense FP16/Int8은 dense 갱신과 fused SDPA.
  - `attend_batched(q, k, v, caches, scale, mask)`: `caches[0]`이 pool 기반이고 mask 없는 한 토큰 단계면 배치 전체를 pooled 호출 한 번으로 처리하고, 거절된 행은 하나씩 `attend`로 보낸다.
- **`MlaLatentCache::attend`**는 absorbed MLA decode 정책을 맡는다. split-KV가 켜져 있고 계획이 받아들이면 split-KV, 아니면 `absorbed_decode`다.
- **모델**: qwen3, llama3, helium은 `attend` / `attend_batched`를 호출하고 배치 forward에서 `DecodeBatchContext` 인자를 뺐다. deepseek_v2의 한 토큰 absorbed decode는 `MlaLatentCache::attend`를 호출한다.
- **ADR 0008**과 문서(`CONTINUOUS_BATCHING.md`, `turbo-kv-cache.md`, `environment-variables.md`, `architecture.md`).
- **테스트**: `attend_tests.rs`(저장소별 분기, Turbo 배치 행과 단일 시퀀스 비교, 빈 배치, 한 행 배치, 저장소가 섞인 배치), MLA `attend`와 압축 해제 기준 구현 비교.

## 3. 기술적 선택과 그 이유

- **`KvStorage` trait 객체 대신 캐시 필드에 대한 match.** 캐시는 이미 양자화 모드를 match로 정한다. 층마다, 토큰마다 가상 호출을 하는 방식은 ADR 0004가 hot loop에서 배제했다.
- **저장소는 캐시를 만들 때 정한다.** `CachePool::allocate_with_layout`은 paged 백엔드의 dense 계열에 FP16 레이아웃일 때만 pool 기반 캐시를 준다. 이 할당이 시퀀스별 저장소 정책이므로, `attend`로 옮긴 계열에서는 `DecodeBatchContext`가 더는 커널을 고르지 않는다. ADR 0007에서 측정으로 정한 기본값(시퀀스가 하나면 dense)은 워커가 `max_batch_size=1`로 도는 곳에서 성립한다. 서버의 `--decode-storage-backend` 정책은 바꾸지 않았고, 배치가 커질 때 시퀀스를 다른 저장소로 옮기는 일은 범위 밖이다.
- **Qwen3와 Llama 3에서 dense 포인터 경로를 제거한다.** 이 경로는 FP16이 아닌 paged 레이아웃이 dense로 남겨 두는 FP16 층(Turbo Boundary-V 층, `--kv-skip-last-layer`일 때 마지막 층)만 처리했다. 제거한 C++ 진입점은 행마다 블록을 잘라 이어 붙인 뒤 SDPA를 부르는 루프였다. 행마다 dense 갱신 후 SDPA를 부르면 같은 attention을 이어 붙이기 복사 없이 계산한다. 이 커널은 Phase 4b 전까지 모델 소유 계열을 위해 남긴다.
- **Turbo 배치의 수치 변화는 받아들인다.** pooled 호출이 거절한 행은 이제 dequant-first 변형을 탄다. 단일 시퀀스 decode가 이미 쓰던 경로다. 배치 출력이 ADR 0007의 '같은 입력, 같은 출력' 불변식에서 멀어지는 것이 아니라 가까워지며, 배치 행이 단일 시퀀스 호출과 비트 단위로 같다는 것을 테스트로 고정했다.
- **배치가 거절된 뒤 행별로 다시 시도하는 동작은 유지한다.** 행마다 pooled 시도를 건너뛰는 변형을 만들면 pool intercept와 gather 경로로 떨어지는데, ADR 0001은 4K 토큰을 넘으면 이 경로가 SDPA보다 2~3배 느리다고 측정했다.

## 4. 검증

- 단위 테스트: mlxcel-core `cache::attend`(11), `mla::`(42), `cache::paged_batch_decode`, `cache::decode_undo`, `cache::paged_detach`. 루트 크레이트는 `models::qwen3::`, `models::llama3::`, `models::helium`, `models::deepseek_v2`, 스케줄러, prompt-cache, 모델 소유 lookahead, `distributed::disaggregated`(189)를 모듈마다 따로 돌렸다.
- clippy(`-D warnings`, 루트와 mlxcel-core 테스트), `cargo fmt --check`, 라이선스 헤더 검사, `vlm_wrapper_capability_delegation`, `dead_doc_pointers`.
- 실제 체크포인트: release `mlxcel-engine-parity`를 `MLXCEL_SDPA_DETERMINISTIC=1`로 변경 전후와 `main`(#2204) 병합 후에 돌렸다. Qwen3-1.7B 4-bit와 Llama-3.2-1B 4-bit 모두 CLI, 서버 dense, 서버 paged 출력이 같았고, 문서화된 prompt-cache 적중 쌍만 이전과 똑같이 갈라졌다.

## 5. 남은 위험

- **Turbo와 paged를 함께 쓰는 배치 서빙은 실제 체크포인트로 돌리지 않았다.** 경계 층 경로와 Turbo 행 경로는 단위 테스트로만 확인했다. epic 마지막 측정에 Turbo paged 배치 처리량을 한 번 넣어야 한다.
- **DeepSeek absorbed decode도 실제 체크포인트로 돌리지 않았다.** 이 경로는 `MLXCEL_MLA_ABSORBED`로 켜야 하고, deepseek-v2-lite는 GB10에서 이미 깨져 있으며, 이번 변경은 기존 호출을 옮긴 것이다. 블록 단위 단위 테스트가 이를 확인한다.
- **기본으로 ignore된 실제 체크포인트 paged 패리티 테스트**(`paged_scheduler_parity`, `serving_handoff_parity_tests`)는 컴파일만 확인했다.
- **모델 소유 계열은 Phase 4b 전까지 `DecodeBatchContext`를 읽는다.**

## 6. 학습 포인트

- **'도달 불가'는 기본 레이아웃 하나가 아니라 모든 레이아웃에서 확인해야 한다.** 첫 초안은 FP16 paged 레이아웃이 pool 기반 캐시를 받으니 dense-compat 경로에 도달할 수 없다고 적었다. 대부분의 층은 양자화하고 일부 층만 FP16로 남기는 혼합 레이아웃은 여전히 그 경로를 탔다.
- **새 진입점은 실제 코드가 호출해야 끝난다.** `MlaLatentCache::attend`는 자체 테스트를 통과했지만 DeepSeek는 여전히 자기 분기를 쓰고 있었다. 리뷰에서 호출 지점을 검색해 찾아냈다.

## 7. 관련 항목

- epic #2166, 이슈 #2171, ADR 0007, ADR 0008.
- #2172(Phase 4b, 미룬 계열), ADR 0001(paged gather 비용), ADR 0002(Turbo dequant-first), ADR 0004(op 단위 동적 분기 배제), #899(pooled paged 진입점), #2182(회전 캐시 decode-undo 로그, 변경 없음).
