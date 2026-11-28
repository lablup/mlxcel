# 기술 보고서: PR #3135 - logit_trace 점수 계산을 행 블록 단위로 나눠 9216 너비 forward가 메모리에 들어가게 함

**작성일**: 2026-11-16
**상태**: 완료
**위험도**: 낮음 (측정 도구만 변경, 모델·런타임 코드는 바뀌지 않음)

## 요약

이슈 #3132는 `main`에서 Llama 4 Scout 9216토큰 단일 forward가 Metal 메모리 부족으로 중단된다고 보고했습니다. 레이어별 피크 메모리를 측정해 보니 원인은 모델 forward가 아니었습니다. lazy로 실행한 9216 너비 forward의 피크는 67.38 GB로, 가중치 60.74 GB보다 6.6 GB 높고 그중 3.3 GB는 반환되는 logits입니다. 중단은 `examples/logit_trace`에서 일어났습니다. `emit_rows_to`가 청크의 모든 행을 202,048개 어휘 전체에 대해 한꺼번에 처리했기 때문입니다(f32 변환, log-softmax, `argpartition`). 이제는 1024행씩 나눠 처리하며, 9216 너비 trace의 피크는 71.18 GB, 16384 너비는 75.48 GB입니다.

## 1. 문제 정의

#3126의 Llama 4 RoPE 측정은 9216 너비 forward 한 번 대신 4608 너비 forward 두 번을 써야 했습니다. 단일 forward가 `kIOGPUCommandBufferCallbackErrorOutOfMemory`로 중단되었기 때문입니다. 당시 남긴 기록은 8192 attention chunk를 넘는 지점에 모델 쪽 메모리 문제가 있음을 시사했습니다. 그게 사실이었다면 trace뿐 아니라 긴 단일 패스 Llama 4 prefill 전체가 영향을 받았을 것입니다.

## 3. 기술적 선택과 그 이유

- **고치기 전에 계측.** 이슈가 꼽은 모델 쪽 용의자는 셋이었습니다. dense chunked mask, 너비에 비례하는 MoE 중간 텐서, prefill 입력을 붙잡는 lazy cache 상태입니다. 레이어별 측정(MLX `get_active_memory` / `get_peak_memory`, `reset_peak_memory`)으로 셋 모두 배제했습니다. 어떤 단계도 직전 살아 있던 메모리보다 0.9 GB 넘게 높아지지 않았고, 레이어마다 늘어난 양은 K/V 38 MB뿐이었습니다. 측정 코드는 임시였고 PR에 포함하지 않았습니다.
- **forward가 아니라 점수 계산을 블록화.** 점수 계산의 모든 단계가 행 단위라서 블록으로 나눠도 출력은 똑같고 피크만 제한됩니다. 1024행 블록이면 너비와 상관없이 logits 위로 4.8 GB만 더 씁니다. 청크 전체를 한 번에 처리하면 4608행에서 22.3 GB, 9216행에서 그 두 배가량이 더 들었습니다.
- **배제한 모델 쪽 특성도 테스트로 고정.** chunk를 넘는 한 행 prefill이 dense chunk mask를 만들지 않음을 확인하는 테스트를 추가했습니다. 배제한 종류의 메모리 증가가 다시 생기면 이 테스트가 잡습니다.

## 7. 변경 요약

| 단계 | 너비 | 피크, main (GB) | 피크, 브랜치 (GB) |
|---|---|---|---|
| `lm_head`까지 하나의 lazy graph로 실행한 forward | 9216 | 67.38 | |
| `lm_head`까지 하나의 lazy graph로 실행한 forward | 16384 | | 72.60 |
| Trace 점수 계산 | 4608 | 85.79 | |
| Trace 점수 계산 | 9216 | 메모리 부족 | 71.18 |
| Trace 점수 계산 | 16384 | | 75.48 |

출력:
- 4608 두 번 trace는 변경 전후 바이트 단위로 동일합니다.
- 9216 너비 forward 한 번 대 4608 너비 forward 두 번: decided 위치 3651개 중 불일치 0개. 전체 top-1 불일치는 0.358%이고 모두 reference 격차 0.44 미만입니다. perplexity는 11.7995 대 11.7967입니다.
- `mlxcel generate --temp 0`에서 8,584 토큰 프롬프트의 64개 출력 토큰이 단일 패스 prefill과 기본 청크 분할에서 같습니다. 15,563 토큰 단일 패스 prefill도 끝까지 실행됩니다.

테스트:
- `blocked_scoring_matches_whole_chunk_scoring`: 1, 2, 4, 6, 1024행 블록 결과를 청크 전체 처리 결과와 비교합니다.
- `a_one_row_prefill_past_the_chunk_builds_no_dense_chunk_mask`: 모델 쪽 특성을 확인합니다.

결과: `docs/benchmark_results/llama4-long-prefill-memory-m5max-2026-11-16.md`.

## 8. 후속 조치

- Llama 4 쪽은 없습니다. 큰 어휘 전체에 대해 청크 전체를 한 번에 처리하는 다른 도구(`examples/perplexity`)도 긴 너비에서 같은 한계에 걸릴 수 있습니다. 이 PR은 그쪽을 바꾸지 않습니다.
