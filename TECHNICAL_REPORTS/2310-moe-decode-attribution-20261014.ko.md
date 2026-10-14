# PR #2310: M = 1 전문가 Gather 구간 MoE 디코드 격차 원인 규명 및 최적화

- **Date**: 2026-10-14
- **Status**: Implemented, reviewed, formatted, and verified
- **Risk**: Low (벤치마크 원인 분석 문서화, Inkling 단일 토큰 퓨즈드 MoE 라우팅 연결, DeepSeek-V4 gather 인덱스 공유)

## Summary

PR #2310은 이슈 #1859에서 제기된 `deepseek-v4-flash-4bit`, `inkling-small-mlx-4bit`, `inkling-small-nvfp4`의 단일 토큰 디코드(`M = 1, B = top_k = 6, E = 256`) 성능 격차 원인을 커널 디스패치 및 그래프 수준에서 규명하여 `docs/benchmark_results/moe-m1-gather-attribution-2026-10-14.md`에 기록하고, 분석 결과에 근거한 두 가지 코드 최적화(`InklingSparseMoe::forward`의 비스케일 affine 체크포인트 단일 토큰 퓨즈드 MoE 커널 연결 및 `src/models/deepseek_v4_moe.rs` 내 `LimitedSwitchGlu::forward`의 gather 인덱스 공유)를 구현합니다.

## Problem

`b8fdddfa` 커밋 기준 Apple Silicon 벤치마크에서 `deepseek-v4-flash-4bit`, `inkling-small-mlx-4bit`, `inkling-small-nvfp4`의 프리필 성능은 `mlx-vlm` 대비 55%~82%인 반면 단일 토큰 디코드 성능은 16%~23%에 머물렀으며, 기존 가설은 `M = 1` 전문가 gather 형상(`x_exp`를 `[n_tokens, top_k, 1, D]`로 확장)이 Metal의 `bm = 32` 타일 `gather_qmm` 또는 CUDA의 `tile_m = 16` `qmm_sm80` 행렬 커널을 호출해 낮은 점유율을 유발한다는 것이었습니다. 이슈 #1859는 `src/models/switch_layers.rs`, `src/models/deepseek_v4_moe.rs`, `src/models/inkling/mlp.rs`를 수정하기 전에 먼저 실제 백엔드 커널 디스패치와 연산 그래프 구성을 규명하도록 요구했습니다.

## Design and Implementation

- **커널 및 연산 그래프 원인 규명 (`docs/benchmark_results/moe-m1-gather-attribution-2026-10-14.md`)**:
  - Metal(`src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/quantized.cpp`), CUDA(`src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/quantized/qmm.cu`), ROCm(`src/lib/mlxcelverse/overlays/rocm/mlx/backend/rocm/quantized/qmm.hip`)의 `GatherQMM::eval_gpu` 디스패치 분기를 추적한 결과, `M = 1, B = 6` 조건에서 세 백엔드 모두 타일 행렬 커널이 아닌 전용 벡터-행렬(QMV) 커널(Metal은 `M = 1 < get_qmv_batch_limit >= 6`으로 `qmv_quad` / `qmv_fast` / `fp_qmv_quad`, CUDA는 `M * B = 6 < 8` 및 `!supports_qmm_sm80(nvfp4)`로 `qmv` / `fp_qmv`, ROCm은 `M = 1 < 4`로 `qmv_gather`)로 분기함을 확인했습니다. 즉 Metal `bm = 32` `gather_qmm`이나 CUDA `tile_m = 16` `qmm_sm80`은 단일 토큰 디코드에서 호출되지 않습니다.
  - `forward_fused_kernel` 연결 모델 패밀리 수를 기준 커밋부터 현재 `main`까지 대조하여 정리했습니다(`b8fdddfa` 시점 17개, `#1739` `laguna_layers.rs` 추가 후 18개, `#2283` `granitemoehybrid.rs` 추가 후 19개, 본 PR의 `inkling/mlp.rs` 연결 후 20개).
  - 모델 패밀리별 실제 디코드 병목을 규명했습니다:
    1. `deepseek-v4-flash-4bit`: 디코드 격차의 85% 이상은 43개 레이어마다 레이어당 2회(토큰당 86회) 실행되는 비융합 20회 반복 `f32` `HyperConnection` Sinkhorn-Knopp 정규화 루프(토큰당 약 12,470개 그래프 노드 생성)에서 기인하며, 추가로 `LimitedSwitchGlu::forward`가 `#2137`의 인덱스 공유 경로를 거치지 않아 매 프로젝션마다 중복 인덱스 합성 노드를 생성하고 있었습니다.
    2. `inkling-small-mlx-4bit`: `InklingSparseMoe::forward`가 `gate_scale`과 `out_scale`이 모두 `None`인 affine 체크포인트에서도 `SwitchGLU::forward_fused_kernel`을 호출하지 않고 항상 `SwitchGLU::forward` + `moe_weighted_sum` 폴백 경로를 실행하고 있었습니다.
- **Inkling 단일 토큰 퓨즈드 MoE 라우팅 (`src/models/inkling/mlp.rs`)**: `flat.shape()[0] == 1 && self.gate_scale.is_none() && self.out_scale.is_none() && fused_moe_enabled()` 조건에서 `self.experts.forward_fused_kernel(&flat, &top_k_indices, &expert_weights)`를 우선 호출하도록 연결하여 희소 MoE 레이어당 GPU 커널 런치 4회와 그래프 노드 10개를 절감했습니다. `gate_scale` / `out_scale`이 존재하는 NVFP4 체크포인트는 기존 `forward_with_expert_scales` 경로를 유지합니다.
- **DeepSeek-V4 Gather 인덱스 공유 (`src/models/deepseek_v4_moe.rs`)**: `LimitedSwitchGlu::forward`가 `expand_dims_multi(x, &[-3, -2])`, `prepare_gather_indices(indices, sorted_indices)`, `SwitchLinear::forward_indexed`를 사용하도록 개선하여 비트 단위 동일성을 유지하면서 레이어당 6개(43개 레이어 기준 토큰당 258개)의 중복 인덱스 그래프 노드를 제거했습니다.

## Review Findings and Fixes

- `pr-reviewer`: CRITICAL/HIGH/MEDIUM 지적 사항 0건으로 통과했습니다.
- `pr-security-checker`: 보안 취약점 및 성능 회귀 0건으로 통과했습니다.
- `pr-finalizer`: 단위 테스트 커버리지, `src/models/switch_layers.rs` 내 `Used by:` 문서 주석 동기화, `cargo fmt --check` 통과를 확인했습니다.

## Verification

- `cargo fmt --all -- --check`: 통과했습니다.
- `MLXCEL_DEVICE=cpu cargo test --profile test-fast --lib models::deepseek_v4::tests::limited_switch_glu -- --test-threads=1`: 1개 통과했습니다.
- `MLXCEL_DEVICE=cpu cargo test --profile test-fast --lib models::inkling::mlp::tests -- --test-threads=1`: 3개 통과했습니다.
- `MLXCEL_DEVICE=cpu cargo test --profile test-fast --lib models::inkling::tests -- --test-threads=1`: 17개 통과했습니다.
- `MLXCEL_DEVICE=cpu cargo test --profile test-fast --lib models::switch_layers::gather_index_tests -- --test-threads=1`: 5개 통과했습니다.
- `cargo clippy --profile test-fast -p mlxcel --lib --tests -- -D warnings`: 경고 0건으로 통과했습니다.
