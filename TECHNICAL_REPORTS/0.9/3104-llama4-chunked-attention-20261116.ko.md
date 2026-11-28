# 기술 보고서: PR #3104 - dense mask 없이 Llama 4 chunked attention 계산

**작성일**: 2026-11-16
**상태**: 완료 (decode 쪽 chunk 경계 버그를 발견해 #3105로 등록)
**위험도**: 중간

## 요약

Llama 4의 MoE(local) 층은 `attention_chunk_size` 블록(llama-4-scout은 8192) 안에서만 causal하게 attention한다. 기존 prefill은 dense `create_chunked_attention_mask`를 하나 만들어 모든 MoE 층이 함께 썼고, #3097 이후로는 이 mask가 Llama 4의 피크를 결정했다(#3100). 이제 padding이 없고 호출자 mask도 없는 prefill에서는 이 층들이 새 `chunked_causal_attention`을 쓴다. 이 함수는 query를 chunk 경계에서 나누고, 각 조각마다 `max(chunk_start, start_position)`부터 그 조각의 마지막 query까지의 key에 대해 MLX causal-mode SDPA를 돌린다. 그래서 chunk 중간에서 시작하는 piece도 같은 chunk에 이미 cache된 key를 본다. M5 Max에서 19,717 토큰 기준 단일 패스 피크가 72.88 GB에서 68.41 GB로 줄었고, greedy 출력은 두 반복 모두 바이트 단위로 같다.

## 3. 기술적 선택과 그 이유

- **dense mask를 유지하는 경우.** `prefill_uses_causal_mode`가 받아들이지 않으면 dense mask를 그대로 쓴다. batched 행, speculative verify 블록, `MLXCEL_PREFILL_CAUSAL_MODE=0`이 여기에 해당한다. `forward_igqa`와 `forward_igqa_rows`는 모두 `Llama4CxxModel::chunked_prefill_mask`로 경로를 고른다.
- **decode와 호출자 mask 단계는 그대로다.** 예전의 plain-`KVCache` forward는 원래 chunk mask를 쓰지 않았다.
- **수치 검증.** teacher-forced logit trace로 dense mask와 새 기본 경로를 비교했다. 7,168 토큰 문맥 뒤에 2048 토큰 forward를 세 번 돌렸고, 각 forward는 중간에 8192 경계를 넘는다. 결과는 top-1 불일치 6,144개 중 0개, decided position 불일치 756개 중 0개, 최대 logit 차이 0.0000이었고, perplexity도 같았다.

## 7. 변경 요약

| 경로 | 피크, main → 브랜치 | 출력 |
|---|---|---|
| 단일 패스 | 72.88 → 68.41 GB | 동일 |
| 2048 청크 | 65.86 → 65.70 GB | 동일 |

prefill 시간 차이는 host 측정 잡음 범위 안이어서 시간 개선은 주장하지 않는다. 새 테스트는 다음 경우에서 새 경로를 dense mask 결과와 비교한다.
- offset 0에서 한 chunk 안, 그리고 여러 chunk에 걸친 경우
- chunk 중간 시작, 그리고 중간에서 시작해 경계를 넘는 piece
- 앞부분이 trim된 cache 뒤, 그리고 정확히 경계에서 시작하는 경우
- f16

이와 별도로 prefill을 나눠 돌린 결과를 단일 패스와 비교하는 모델 테스트도 있다.

## 8. 후속 조치

#3105: decode에서 chunked 층이 mask 없이 마지막 8192개 key에 attention하므로, 위치 8192를 넘으면 이전 chunk의 key까지 포함한다. mlx-lm은 매 단계 `block_pos == 0` chunk mask를 다시 만든다. 이 버그는 이 PR 이전부터 있었다.
