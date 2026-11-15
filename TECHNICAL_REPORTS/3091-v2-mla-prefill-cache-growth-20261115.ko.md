# 기술 보고서: PR #3091 - DeepSeek V2 prefill 중 대체된 KV buffer를 해제

**작성일**: 2026-11-15
**상태**: 완료 (남은 차이는 V2의 decompressed cache 구조 때문)
**위험도**: 중간

## 요약

DeepSeek V2의 기본 MLA prefill은 2048 토큰 piece에서 absorbed 경로보다 피크가 12.85 GB 높았다(#3084). 층별로 분해해 보니 원인은 두 가지였다. 첫째(약 7.0 GB)는 기본 경로가 head별 K와 V를 decompressed 형태로 저장하는 것으로, 설계상 의도된 부분이다. 둘째(약 6.2 GB)는 dead buffer다. piece가 층의 cache를 키우면 이전 buffer는 MLX allocator cache로 돌아가는데, 같은 piece 안의 이후 요청은 모두 더 큰 크기를 요구하므로 그 buffer를 다시 쓰지 못한다. 이제 `KVCache::reserve_rows`가 층의 graph를 만들기 전에 그 층의 buffer를 키우고, 복사를 `async_eval`로 예약하며, 다음 층을 reserve할 때 allocator cache를 비운다. M5 Max에서 28,705 토큰 프롬프트 기준 2048 piece 피크가 27.98 GB에서 21.04 GB로 줄었고, greedy 출력은 바이트 단위로 그대로다.

## 1. 문제 정의

이슈에서는 piece마다 latent cache를 다시 펼친다고 가정했지만, 측정 결과는 그렇지 않았다. 마지막 2048 토큰 piece의 27개 층을 지나는 동안 allocator cache가 3.05 GB에서 10.47 GB로 늘었고, engine은 이 cache를 piece가 끝난 뒤에만 비운다(#672).

## 3. 기술적 선택과 그 이유

- **reserve는 한꺼번에가 아니라 층마다 한다.** forward를 시작할 때 모든 층을 한꺼번에 reserve하면 모든 층의 이전 buffer와 새 buffer를 동시에 들고 있게 된다. 이 방식은 27.97 GB로 측정되어 채택하지 않았다.
- **같은 증가 규칙을 쓴다.** `update_fp16`도 같은 growth helper를 쓰므로, 키운 buffer는 graph 안에서 키우던 기존 방식의 결과와 바이트 단위로 같다.
- **적용 범위.** decode 행, 첫 fill, 공간이 충분한 append, quantized 또는 paged 구조에서는 reserve가 아무 일도 하지 않는다. DeepSeek V2는 decompressed cache와 latent cache를 갱신하기 전에 각각 reserve를 호출한다.
- **비용.** 바이너리 하나로 두 모드를 측정했을 때 prefill이 약 2% 느려졌다(off 41.97, 42.15 s / on 43.13, 42.87 s).

## 7. 변경 요약

| 셀 | 피크, main | 피크, 브랜치 | 출력 |
|---|---|---|---|
| 기본, 2048 piece | 27.98 GB | 21.04 GB | 동일 |
| `MLXCEL_MLA_ABSORBED=1`, 2048 piece | 15.12 GB | 14.35 GB | 동일 |
| 기본, 단일 패스 | 54.44 GB | 54.43 GB | 동일 |

테스트:
- 새 `cache::fp16_growth_tests` 3개.
- `cache::kv_trim_tests`, `mla::`, `models::deepseek_v2`가 통과하고, #3090 위로 rebase한 뒤 last-row 테스트도 통과한다.

자세한 내용: `docs/benchmark_results/mla-v2-prefill-cache-growth-m5max-2026-11-15.md`.

## 8. 후속 조치

- chunked prefill 중 dense FP16 cache가 커지는 모델은 모두 같은 dead buffer를 남길 것이므로, 다른 모델도 `reserve_rows`를 도입할 수 있다.
- 첫 fill에서는 cache가 커지지 않으므로 단일 패스 피크는 변하지 않는다.
