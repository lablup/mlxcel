# PR #2138: Metal `gather_qmm_rhs`의 행 수를 broadcast된 x 기준으로 지정

**작성일**: 2026-10-06
**상태**: Draft. 근본 원인은 소스로 입증. 수정은 Metal에서 미검증 (Metal overlay가 컴파일되지 않는 CUDA 호스트에서 작성)
**위험도**: 낮음 (모든 프로덕션 호출에서 no-op이며, `gather_qmm_rhs`가 x를 broadcast해야 하는 경우에만 동작이 바뀜)

## 요약

M1 Ultra 러너의 Metal nightly는 2026-09-30부터 `models::switch_layers::mxfp_tests::{mxfp4,mxfp8}_gather_qmm_matches_host_reference`의 `SortedSharedActivation` 케이스에서 실패했다. MLX Metal 백엔드의 `GatherQMM::eval_gpu`는 `gather_qmm_rhs`에 행 수로 `x.size() / K`를 넘긴다. 이 값은 `gather_qmm_rhs`가 x를 indices에 맞춰 broadcast하기 전의 행 수다. 활성 행 하나를 정렬된 슬롯 32개에 gather하면 이 값은 1이 되어, grid와 커널의 `M` 경계가 한 행만 덮고 출력의 1..31행은 기록되지 않는다. 이 PR은 overlay에서 `B * M`을 넘긴다.

Refs #1599.

## 1. Nightly 이력

| 날짜 | 실패 | 상태 |
|---|---|---|
| 2026-09-28 | `verify-clippy`: `fused_norm_parity_tests.rs:158`, `fused_rope_parity_tests.rs:133`의 미사용 `Result` | #2029 (d8d34e2b)에서 수정 |
| 2026-09-29 | `tests::family_order_is_exhaustive`: `FAMILY_ORDER`에 `Speech` 누락 | #2080 (5486e404)에서 수정 |
| 2026-09-30 ~ 10-05 | mxfp gather 테스트 두 개 | 이 PR |

mxfp 테스트는 2026-09-30 #2071 (2cee9cf4)과 함께 추가되었으므로, 마지막 실패는 회귀가 아니라 새 테스트가 upstream의 잠재된 경계 사례를 드러낸 것이다.

## 2. 근본 원인

- 케이스 입력은 x `[1, 1, K]`, expert 8개에 대한 indices `[32]`, `sorted = true`다. M == 1, B = 32, B / E = 4, `right_sorted_`이므로 `eval_gpu`는 `gather_qmm_rhs`로 간다.
- `gather_qmm_rhs`는 x를 32행으로 broadcast하지만(`broadcast_with_indices`), `M` 인자는 호출자가 원래 x로 계산한 1이다. `grid_dims.y = ceil(M / 16)`과 커널의 `tgp_bm = min(BM, M - y_row)` 때문에 0행만 기록된다. `out`은 `allocator::malloc`에서 오므로 나머지 행에는 이전 메모리 값이 남는다: mxfp4는 상대 오차 7.4e32, mxfp8은 non-finite.
- 같은 실행에서 앞선 gather 케이스 세 개는 bf16, f16, f32 모두 통과했고, MLX CPU 백엔드에서 `SortedSharedActivation`을 돌리는 `mxfp_matmuls_match_host_reference_on_cpu_device`도 통과했다. reference와 테스트는 올바르며, 메시지의 "bf16"은 처음 시도한 dtype일 뿐이다.
- 결함은 모드와 무관하고(affine도 동일), ml-explore/mlx main에도 그대로 있다.

## 3. 변경

유일한 호출 지점에서 `x.size() / K`를 `B * M`(출력 행 수)으로 바꿨다. x가 이미 확장된 경우 기존 값과 같으며, 모든 프로덕션 호출(`SwitchLinear::forward(.., true)`는 항상 `gather_sort` 출력을 받음)이 이에 해당한다. 같은 인자를 받는 NAX 변형도 함께 고쳐진다. overlay 헤더와 CMake overlay 주석에 새 hunk를 기록했다.

## 4. 검증

- CUDA (GB10): 회귀 없음. 실행 결과는 PR 본문 참조.
- Metal: 미실행. Mac에서 `cargo test --profile test-fast --features metal,accelerate -p mlxcel --lib models::switch_layers::mxfp_tests -- --test-threads=1`과 슬롯 64개 이상의 MoE prefill 스모크가 필요하다.

## 5. 후속 작업

ml-explore/mlx에 이 호출을 보고하고, upstream 수정이 포함된 pin bump 때 이 hunk를 제거한다.
