# 기술 보고서: PR #2132 - TF32 테스트 고정값 감시 테스트와 디코드 영향 기록 (Issue #1065)

**날짜**: 2026-10-06

**상태**: GB10에서 구현 및 검증 완료, 머지 대기.

**언어**: Rust (테스트와 문서 주석), Markdown

**위험도**: 낮음 (런타임 코드 변경 없음)

## 요약

Issue #1065는 M5 Max에서 f32 수치 일치 테스트 네 개가 fp16 엡실론 수준 오차로 실패한다고 보고했고, 이를 Apple 17세대 결함으로 규정했습니다. 이후 코멘트에서 원인이 MLX의 기본값 `MLX_ENABLE_TF32=1`이며 CUDA에서도 재현된다는 점이 확인되었습니다. #1260은 두 lib 테스트 바이너리에서 이 변수를 0으로 고정했습니다. 이 PR은 그 고정값을 감시하는 테스트를 추가하고, 문제를 Apple 전용으로 설명하던 문서 주석을 바로잡고, 남은 chunked SDPA 단언 두 개가 측정 오차를 출력하도록 하며, 디코드 A/B 측정 결과와 함께 변수를 문서화합니다.

## 메커니즘

MLX는 두 백엔드에서 정밀도를 낮춘 f32 GEMM을 선택합니다. CUDA에서는 cuBLAS `CUBLAS_COMPUTE_32F_FAST_TF32`(`gemms/cublas_gemm.cpp`), Apple 17세대 GPU에서는 NAX 커널(`is_nax_available() && (enable_tf32() || dtype != float32)`)입니다. CUDA의 `can_use_gemv`는 전치된 가중치에 대한 `M == 1`(및 `N == 1`)을 이 설정과 무관한 gemv 커널로 보내며, SDPA는 별도 커널을 씁니다. 그래서 #1065가 지목한 `mlxcel-core` 테스트 네 개는 CUDA에서 `MLX_ENABLE_TF32=1`이어도 통과하고, 루트 lib 테스트 세 개(bailing GLA, deepseek_v2 MLA, florence2)는 실패합니다.

## 감시 테스트

`tf32_pin_tests::f32_gemm_runs_at_full_precision_in_the_test_process`는 16x512와 512x16 f32 행렬을 곱하고(두 백엔드 모두 실제 GEMM 경로) f64 호스트 합과 절대 오차 1e-4 기준으로 비교합니다. GB10 결과: 기본값과 `=0`에서 통과, `=1`에서 3/3 실패(5.84e-3).

## 디코드 A/B (GB10, greedy, 256 토큰)

| 체크포인트 | 가중치 | 결과 |
|---|---|---|
| plamo-2-1b | f32 | 프롬프트 5개 중 1개가 약 40단어 이후 갈라짐, 각 조건은 5회 모두 결정적, 처리량 양쪽 38 tok/s |
| llama-3.1-8b-bf16 | f16 | 동일, 프롬프트 3개 x 2회 |
| qwen2.5-0.5b-bf16 | bf16 | 양쪽 조건 모두 실행 간 변동, TF32로 귀속 불가 |
| qwen3-0.6b | 4-bit | 양쪽 조건 모두 실행 간 변동, TF32로 귀속 불가 |

결정: 운영 기본값은 MLX 기본값을 유지합니다. TF32는 dense f32 GEMM에만 영향을 주며 실제로는 f32 체크포인트에 해당합니다(CUDA에서는 bias 없는 단일 행 matmul이 gemv로 가므로 주로 prefill 크기의 GEMM이 대상이며, 이는 측정이 아니라 MLX 소스에서 읽은 내용입니다). f32와 정확히 같은 출력이 필요하면 `MLX_ENABLE_TF32=0`을 쓰도록 문서에 안내했습니다.

## 미검증 항목

이 호스트에는 Metal 하드웨어가 없습니다: M5 및 M1 Ultra 재실행, M5에서 감시 테스트의 실패 조건, M5 대 M1 디코드 품질.
