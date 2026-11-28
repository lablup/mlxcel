# 기술 보고서: PR #3096 - 모든 모델의 prefill에서 대체된 FP16 KV buffer를 해제

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 중간 (dense cache를 쓰는 모든 모델이 공유하는 경로)

## 요약

청크 prefill의 한 piece가 층의 dense FP16 `KVCache`를 키우면, 이전 buffer는 MLX allocator cache에 남는다. 같은 piece 안에서 이후에 오는 요청은 모두 더 큰 크기를 요구하므로 이 buffer를 다시 쓰지 못하고, engine은 piece가 끝날 때만 cache를 비운다. #3091은 이 문제를 DeepSeek V2에서만 고쳤다. 이제는 `KVCache::update_fp16` 안의 공용 증가 경로에서 buffer를 해제하므로, dense FP16 cache를 쓰는 모든 모델에 적용된다(#3092). M5 Max에서 28K 토큰 프롬프트를 2048 토큰 piece로 돌리면 피크가 qwen3-4b는 10.94 GB에서 7.26 GB로, llama-3.1-8b는 10.39 GB에서 8.09 GB로, gemma-3-4b는 5.25 GB에서 4.86 GB로 줄었다. greedy 출력은 바이트 단위로 같고, prefill 시간 변화는 3% 이내다. 이제 세 모델 모두 2048 토큰 piece의 피크가 단일 패스보다 낮다.

## 3. 기술적 선택과 그 이유

- **경로는 공용이고 동작 조건은 좁다.** 해제는 기존 buffer보다 커지는 다중 토큰 append에서만 일어난다. 그때 allocator cache를 비우고, 이전과 같은 규칙으로 키운 buffer를 만들고, 복사는 `async_eval`로 예약한다. 첫 fill, decode 행, 공간 안에 들어가는 append, quantized·rotating·pool 기반 구조는 건드리지 않는다.
- **DeepSeek V2의 명시적 호출을 없앴다.** 두 cache 모두 `update_fp16`을 거치므로 `reserve_rows`를 삭제했고, 피크는 main과 같다.
- **메모리 검사는 대조군과 비교한다.** piece가 쓰는 K/V 입력도 allocator cache에 남기 때문에, 통합 테스트(`tests/kv_growth_release.rs`)는 고정 임계값 대신 지연 증가 대조군과 비교한다(151 MB 대 403 MB). 해제를 일부러 끄면 이 테스트도 실패한다.

## 7. 변경 요약

| 모델 | 피크, main | 피크, 브랜치 | prefill | 출력 |
|---|---|---|---|---|
| qwen3-4b-4bit | 10.94 GB | 7.26 GB | -1.1% | 동일 |
| meta-llama-3.1-8b-instruct-4bit | 10.39 GB | 8.09 GB | +2.9% | 동일 |
| gemma-3-4b-it-4bit (34개 층 중 5개가 `KVCache` 사용) | 5.25 GB | 4.86 GB | +2.7% | 동일 |
| deepseek-v2-lite-chat-4bit-mlx | 20.20 GB | 20.20 GB | +0.2% | 동일 |

테스트: `mlxcel-core` lib suite(2308), retention 바이너리 2개, 직렬로 돌린 전체 `mlxcel` lib suite(11,950)가 통과한다. 자세한 내용: `docs/benchmark_results/kv-growth-release-all-models-m5max-2026-11-15.md`.
