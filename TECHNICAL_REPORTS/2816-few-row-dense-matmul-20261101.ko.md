# PR #2816: M1 Ultra 인수 시험에서 찾은 배치 디코드 저하 수정

**작성일**: 2026-11-01
**상태**: 완료
**언어**: C++, Rust
**위험도**: 중간 (M1, M2의 연산 경로 변경)

## 요약

#2587 인수 시험에서 M1 Ultra의 B=2, B=4 저하가 재현되었습니다. GPU 세대 13, 14에서 2~6행 조밀 곱은 GEMV를 유지하고, `SwitchGLU::forward`는 1토큰이거나 M3 이상일 때만 gate/up 커널을 씁니다.

## 1. 문제 정의

세대 13, 14는 `gemv_wide`에서 제외되어, 2행 이상 조밀 곱이 1행 GEMV의 몇 배 비용인 steel GEMM 타일로 떨어졌습니다. pythia-1b는 B=2에서 165 tok/s로 B=1보다 느렸습니다. 별도로, M3 Ultra에서만 측정된 #2514의 gate/up 커널은 M1 Ultra에서 `gather_qmm`보다 약 10% 느렸습니다.

## 7. 변경 요약

| 영역 | 요약 |
|---|---|
| Metal | 새 matmul 오버레이, `MLXCEL_DENSE_GEMV_ROWS` |
| MoE | `switch_gate_up_preferred_in_forward` |
| 근거 | `docs/benchmark_results/versions/v0.8.0.md#few-row-dense-matmul-m1ultra-2026-11-01` |

## 8. 후속 조치

남은 차이와 기본 M1, M2 측정은 #2817에서 추적합니다.
