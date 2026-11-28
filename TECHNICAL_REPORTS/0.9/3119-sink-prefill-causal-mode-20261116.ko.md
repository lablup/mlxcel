# 기술 보고서: PR #3119 - attention sink prefill을 MLX causal/window mode로 실행

**작성일**: 2026-11-16
**상태**: 완료 (MiMo v2 Flash와 DeepSeek-V4는 로컬 checkpoint가 없어 unit test로만 확인)
**위험도**: 중간

## 요약

#3056과 #3097에서 일반 causal prefill과 sliding window prefill은 dense mask 배열 대신 MLX causal mode를 쓰게 됐다. 하지만 attention sink가 있는 층은 sink를 받는 causal mode SDPA 진입점이 bridge에 없어서, 층마다 (L, L) band를 계속 만들고 있었다(#3098). 이제 bridge에 `fast_scaled_dot_product_attention_with_sinks_causal`이 있고, 새 `prefill_sdpa::causal_prefill_attention`이 MLX probe가 sink 변형이 fused된다고 답할 때 이 진입점으로 보낸다. 그렇지 않거나 새 진입점이 오류를 반환하면 band 방식으로 되돌아간다. GPT-OSS, Falcon-OCR, MiMo v2 Flash, 전부 local 층으로 된 DeepSeek-V4 stack, Laguna가 이 경로를 쓴다.

M5 Max에서 단일 패스 피크가 줄었다.

| 모델 | 토큰 수 | 피크 | prefill 시간 |
|---|---|---|---|
| gpt-oss-20b | 19,773 | 22.54 → 17.86 GB | 6.63 → 3.62초 |
| laguna-xs.2 | 28,747 | 35.84 → 29.25 GB | 보고 안 함 |
| falcon-ocr | 9,067 | 5.11 → 4.70 GB | 보고 안 함 |

모든 측정 칸에서 출력이 같다.

## 3. 기술적 선택과 그 이유

- **sink 여부를 포함해 probe한다.** `sdpa_uses_fused_kernel`이 `has_sinks`를 받고 memo key에도 넣는다. 그래서 sink가 없으면 fused되지만 sink가 있으면 안 되는 shape는 계속 band를 쓴다.
- **이번 범위 밖.**
  - OpenAI privacy filter는 대칭 양방향 band를 써서 causal mode로 표현할 수 없다.
  - DeepSeek-V4의 compressed/sparse 층은 local mask에 pooled row를 덧붙인다. 그래서 이런 층이 하나라도 있는 stack은 mask 배열을 유지한다.
- **목록 문장은 이제 `create_causal_mask` 파일 13개로 적혀 있다.**

## 7. 변경 요약

| 모델 | 경로 | 피크, main → 브랜치 | prefill, main → 브랜치 |
|---|---|---|---|
| gpt-oss-20b-mxfp4-q4 | 단일 | 22.54 → 17.86 GB | 6.63 → 3.62 s |
| gpt-oss-20b-mxfp4-q4 | 2048 청크 | 13.42 → 13.16 GB | 3.96 → 3.84 s |
| laguna-xs.2-4bit | 단일 | 35.84 → 29.25 GB | 39.03 → 7.32 s (main 값은 다른 작업이 함께 돌던 호스트에서 1회 측정) |
| laguna-xs.2-4bit | 2048 청크 | 22.32 → 21.99 GB | 9.71 → 7.49 s |
| falcon-ocr | 단일 | 5.11 → 4.70 GB | 0.79/0.94 → 0.59/0.28 s |

테스트: 새 `prefill_sdpa_tests`는 sink 진입점과 경로 선택 helper의 결과를 명시적 mask 호출 및 f64 reference와 비교한다. f32와 f16에서, 여러 window, head dim, GQA 설정으로 확인한다. 잘못된 입력은 `Err`를 반환한다. `mlxcel-core` lib suite(2,324개)와 전체 `mlxcel` lib suite(11,976개)가 통과했고, workspace clippy도 경고가 없다.
