# 기술 보고서: PR #3097 - 남은 prefill mask에도 MLX causal mode 적용

**작성일**: 2026-11-15
**상태**: 완료 (sink SDPA와 mask를 직접 더하는 MLA 경로는 의도적으로 array mask 유지)
**위험도**: 중간 (모델 파일 31개, tensor-parallel 경로와 pipeline 경로)

## 요약

#3056은 Qwen hybrid 계열과 Qwen-VL에서 순수 causal prefill을 MLX causal mode로 돌렸다. 이 PR은 #3072의 나머지를 처리한다.
- **순수 causal decoder**는 이제 `prefill_causal_mask`를 쓴다. falcon_h1, plamo2, lfm2, bailing_moe_linear, llama4 global 층, mistral4, minicpm3, nemotron_nas, GLM4V 계열, hunyuan_vl, paddleocr_vl, ernie4_5_moe_vl, TP Qwen 3.5가 해당한다.
- **sliding hybrid**는 새 `prefill_sliding_window_mask`를 쓴다. window 안쪽에서는 causal mode를, window를 넘어서면 MLX의 native `window_size`를 쓴다. cohere2, cohere2_moe, cohere_compass, olmo3, ministral3, exaone4, exaone_moe, mellum, step3p5, afmoe, gemma3n이 해당한다.
- **Gemma 2의 softcap attention**은 mask를 층마다가 아니라 forward마다 한 번만 만든다.
- **maskless padded prefill**에 다섯 계열이 더 opt-in한다.

M5 Max에서 단일 패스 피크가 gemma-3n-e2b는 23.05 GB에서 13.05 GB로, exaone-4.0-1.2b는 9.82 GB에서 4.05 GB로 줄었고, prefill은 2.1~2.3배 빨라졌다. greedy 출력은 바이트 단위로 같고, teacher-forced trace에서도 6개 모델 각각 6144개 위치 중 불일치가 0개다.

## 1. 문제 정의

이 모델들은 dense f32 `[L, L + offset]` causal 또는 sliding mask를 forward마다 한 번 만들어 모든 층에 넘겼다. 28K 단일 패스에서 이 mask는 3.3 GB다. 또 array mask를 쓰면, MLX가 causal mode라면 fuse했을 SDPA도 fused kernel을 쓰지 못한다.

## 3. 기술적 선택과 그 이유

- **causal fallback이 없던 곳에 추가했다.** 이슈는 mask가 `None`이면 causal attention으로 넘어간다고 가정했지만, falcon_h1, plamo2, nemotron_nas, mistral4, minicpm3, GLM4V 계열의 attention은 mask 없는 SDPA를 그대로 돌리고 있었다. 새 `mlxcel_core::maskless_prefill_attention`이 이들에 causal fallback을 주고, key가 window보다 길어진 뒤에만 window를 넘긴다.
- **층별 window 필드를 두었다.** ministral3의 full-attention 층은 `window_size`에 sliding window 값을 갖고 있다. 그래서 ministral3과 afmoe에 층별 `prefill_window` 필드를 추가해 full-attention 층에 window가 적용되지 않게 했다.
- **분산 경로를 단일 프로세스 모델과 맞췄다.** Jamba와 Nemotron-H의 pipeline stage, TP Gemma 3는 단일 프로세스 모델처럼 mask 없이 돈다. pipeline의 GLM4-MoE-Lite stage는 MLA이고 단일 프로세스 모델도 array mask를 만들기 때문에 그대로 두었다.
- **범위 밖:**
  - sink SDPA (gpt_oss, falcon_ocr, laguna, mimo_v2_flash, deepseek_v4)
  - mask를 직접 더하는 MLA 경로 (kimi_linear, glm4_moe_lite, kimi_k3, longcat_flash_ngram)
  - speculative verify 블록

## 7. 변경 요약

| 모델 | 경로 | 피크, main → 브랜치 | prefill, main → 브랜치 | 출력 |
|---|---|---|---|---|
| gemma-3n-e2b-it-4bit | 단일 / 2048 | 23.05 → 13.05 / 7.65 → 6.82 GB | 10.48 → 4.53 / 7.19 → 4.74 s | 동일 |
| exaone-4.0-1.2b-4bit | 단일 / 2048 | 9.82 → 4.05 / 5.15 → 4.72 GB | 6.07 → 2.87 / 3.75 → 3.08 s | 동일 |
| lfm2-8b-a1b-4bit | 단일 / 2048 | 13.50 → 11.63 / 8.37 → 8.25 GB | 4.06 → 2.06 / 2.40 → 2.22 s | 동일 |
| c4ai-command-r7b (8K) | 단일 / 2048 | 8.57 → 7.99 / 7.46 → 7.31 GB | 2.30 → 2.06 / 2.09 → 2.06 s | 동일 |
| glm-4.1v-9b (text) | 단일 / 2048 | 18.10 → 18.08 / 8.91 → 8.40 GB | 변화 없음 / 11.94 → 11.16 s | 동일 |
| llama-4-scout-17b-16e | 단일 / 2048 | 72.88 → 73.00 / 66.85 → 66.79 GB | 27.50 → 26.07 / 19.64 → 19.41 s | 동일 |

테스트:
- 새 prefill sliding-mask 테스트 9개.
- `mlxcel-core` lib suite(2317)와 직렬로 돌린 전체 `mlxcel` lib suite(11,950)가 통과한다.
- cohere2, olmo3, minicpm3, gemma3n, mistral4의 maskless padded-prefill parity 검사가 통과한다.
- workspace clippy에 경고가 없다.

자세한 내용: `docs/benchmark_results/prefill-causal-mode-remaining-m5max-2026-11-15.md`.

## 8. 후속 조치

- Llama 4의 피크는 여전히 MoE 층의 chunked-attention mask가 결정한다.
- GLM4V 계열 VLM(MRoPE state), nemotron_nas(모델이 직접 관리하는 cache), Cohere2-MoE, EXAONE-MoE는 maskless padded prefill에 opt-in하지 않았다. 앞의 둘은 위 이유 때문이고, 뒤의 둘은 검증할 로컬 checkpoint가 없어서다.
