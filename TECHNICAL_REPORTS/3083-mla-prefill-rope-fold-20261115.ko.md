# 기술 보고서: PR #3083 - 다중 토큰 단계에서 MLA rope score를 SDPA에 합침

**작성일**: 2026-11-15
**상태**: 완료 (DeepSeek V2 기본 경로의 피크는 별도 조사 필요)
**위험도**: 중간

## 요약

DeepSeek-V3 계열의 MLA prefill은 층마다 `q_pe @ k_peᵀ`를 dense `[B, H, L, L + offset]` 텐서로 계산하고, 여기에 causal mask를 f32로 더한 합을 SDPA의 additive mask로 넘겼다. 새 `mla::rope_folded_attention`은 `[q_nope | q_pe]`와 `[k | k_pe]`를 head 차원으로 이어 붙여 SDPA가 전체 score를 직접 계산하게 한다. mask는 causal mode, 호출자가 준 mask, V3.2의 행별 top-k mask 중 하나가 된다. M5 Max에서 kimi-vl-a3b의 피크는 단일 패스 76.31 GB에서 32.03 GB로, 2048 토큰 청크 20.0 GB에서 14.43 GB로 줄었고, prefill은 8~24% 빨라졌다.

## 1. 문제 정의

다중 토큰 경로의 각 층이 전체 query 길이 × cache 길이 크기의 score 텐서를 따로 만들었다. 이것이 Kimi-VL, Kimi-K2, Youtu의 prefill에서 가장 큰 메모리 항목이었다(#3071).

## 3. 기술적 선택과 그 이유

- **mask를 더하지 않고 이어 붙인다.** mlx-lm의 `deepseek_v3.py`와 같은 방식이다. query 폭(192, absorbed 576)이 value 폭(128, absorbed 512)과 달라 MLX는 이 shape를 fuse하지 않는다. 따라서 호출은 MLX fallback으로 가고, #3049 gate가 이를 Metal 256 MiB budget 안에서 query 축으로 나눈다. additive 텐서는 사라지고 fallback 작업 메모리는 상한이 있다.
- **decode는 그대로다.** 단일 토큰 decode는 기존 rope-mask 형태를 유지한다.
- **적용 범위:** `deepseek_v3`, `deepseek_v32`(V3.2의 top-k mask는 모든 head가 공유하는 `[B, 1, L, S]`가 된다), 그리고 `MLXCEL_MLA_ABSORBED=1`일 때 DeepSeek V2가 쓰는 공용 absorbed 경로.
- **연산 순서가 바뀐다.** Kimi-VL 단일 패스의 greedy 텍스트는 근소한 차이로 갈리는 단어 선택 하나에서 main과 달라진다. teacher-forced logit trace로 보면 decided position 불일치가 cold start에서 2627개 중 1개(0.038%), 앞에 2048 토큰 문맥이 있을 때 2731개 중 0개다. perplexity는 브랜치가 0.04~0.06% 낮다. 2048 청크 셀은 모두 바이트 단위로 동일하다.

## 7. 변경 요약

| 모델 | prefill | 피크, main | 피크, 브랜치 | prefill, main / 브랜치 |
|---|---|---|---|---|
| kimi-vl-a3b-thinking-4bit | 2048 청크 | 19.33-19.99 GB | 14.43 GB | 28.07 / 25.78 s |
| kimi-vl-a3b-thinking-4bit | 단일 패스 | 76.31 GB | 32.03 GB | 35.37 / 30.88 s |
| deepseek-v2-lite-chat-4bit-mlx | 2048 청크 | 27.98 GB | 27.98 GB | 42.84 / 42.81 s |
| deepseek-v2-lite, `MLXCEL_MLA_ABSORBED=1` | 2048 청크 | 15.13 GB | 15.13 GB | 42.73 / 42.64 s |

`mla::` 테스트(53개, 접힌 호출을 f64 host 기준값과 비교하는 새 테스트 5개 포함)와 수정한 모델 모듈의 테스트가 통과한다. 자세한 내용은 `docs/benchmark_results/mla-prefill-rope-fold-m5max-2026-11-15.md`에 있다.

## 8. 후속 조치

- DeepSeek V2-Lite의 기본(non-absorbed) 경로는 2048 청크에서 absorbed 경로보다 피크가 12.85 GB 높다. 기본 경로는 이미 이어 붙이고 있었으므로 score 텐서가 원인이 아니며, 별도 조사가 필요하다.
- Kimi-VL 단일 패스에 남은 피크는 대부분 모든 위치의 logits(#3070)다.
- 이 host에 맞는 checkpoint가 없어 DeepSeek V3.2는 unit test로만 확인했다.
