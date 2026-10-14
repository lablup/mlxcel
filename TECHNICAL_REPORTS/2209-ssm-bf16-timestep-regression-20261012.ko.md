# 기술 보고서: PR #2209 - SSM BF16 timestep 회귀 테스트

**날짜**: 2026-10-12

**상태**: 로컬 CUDA runtime 및 Clippy 검증 완료

**언어**: Rust

**위험 수준**: 낮음

## 요약

PR #2209는 CUDA SSM parity 실패를 진단하고 compiled timestep 계산 전에 `dt_bias`를 float32로 변환했습니다. 이후 PR #2270이 동일한 production 수정을 `main`에 반영했으므로, 현재 `main`을 병합한 최종 PR은 기여자의 원래 이력을 보존하면서 정밀도 계약을 독립적으로 검증하는 회귀 테스트를 추가합니다.

회귀 테스트는 recurrence를 해석적으로 계산할 수 있는 값으로 단순화하고, float32 및 BF16 activation과 BF16-to-float32-to-BF16 bias 전환을 검사하며, 기존 float32 state tolerance를 그대로 유지합니다. Parent parity file이 저장소 workflow의 500줄 제한 아래에 머물도록 테스트는 child module에 배치했습니다.

## 1. 문제 정의

`ssm_update_parity_tests`의 Granite BF16 case는 compiled preprocessing이 `dt`만 float32로 변환하고 `dt_bias`는 BF16으로 남겨 두었기 때문에 실패했습니다. MLX는 mixed-type 덧셈 결과로 BF16을 선택했고, backend recurrence kernel이 값을 받기 전에 `softplus(dt + dt_bias)`가 반올림되었습니다. 이후 state를 float32로 저장해도 이미 손실된 정밀도를 복구할 수 없었습니다.

CUDA kernel body와 float32 MLX reference는 올바랐습니다. 결함은 backend dispatch 전의 공유 `compute_dt_compiled` helper에 있었습니다. Mainline PR #2270은 이미 두 operand를 float32로 변환하므로, 이 PR은 해당 동작을 회귀하지 않도록 고정합니다.

## 2. 변경 요약

새 case는 초기 state를 0으로 두고 `x = B = 1`, `C = D = 0`, `A_log = 0`으로 설정합니다. 따라서 모든 next-state 원소는 처리된 timestep `softplus(-2.5)`와 같습니다. 두 입력은 float32와 BF16에서 정확히 표현되며 expected value는 host float32 연산으로 계산하므로, oracle은 다른 GPU op나 입력 quantization 오차에 의존하지 않습니다.

Case는 float32 및 BF16 activation을 실행하면서 저장된 bias를 BF16, float32, BF16 순서로 바꿉니다. 이를 통해 기존 output/state dtype assertion과 tolerance를 유지하면서 정밀도 계약과 compilation cache의 dtype 전환을 함께 검사합니다.

## 3. 기술적 선택과 그 이유

### 3.1 Mainline production code 유지

PR #2270이 동일한 명시적 float32 cast와 측정 근거를 이미 반영했으므로 `compute_dt_compiled` merge conflict는 현재 `main` 구현으로 해결했습니다. 최종 diff는 production 변경을 중복 주장하지 않으며, 원래 PR commit은 history에 그대로 남습니다.

### 3.2 독립 scalar oracle 사용

Fused kernel을 다른 MLX graph와만 비교하면 같은 promotion 동작을 반복할 수 있습니다. 단순화한 recurrence와 host float32 expected value는 timestep narrowing을 직접 식별하고 tolerance를 넓히지 않은 채 기존 mixed-type preprocessing에서 실패합니다.

## 4. 검증

| 검사 | 결과 |
|---|---|
| 게시된 PR baseline | `33265bb76c0244b66d198e51667256b15ec33e1a` |
| 안정화한 source candidate | `a9696bd8cb3d0550d312ce5b9c561c544f9886e4` |
| 집중 CUDA no-run build | 두 build job과 `-p mlxcel-core --profile test-fast --features cuda --lib --no-run`으로 통과 |
| Rust formatting 및 diff whitespace | 통과 |
| 집중 GB10 SSM runtime | `gpu-lock` 아래에서 6 passed, 0 failed, 0 ignored, 여섯 case 모두 9.65초에 실행 |
| 저장소 contract gate | crate version 일치, kernel dtype/cache-key 검사, kernel-port dispatch, llama-server compatibility manifest, ROCm overlay record, Python tooling test 123개, WebUI fixture/DTO/schema 검사 49개 통과 |
| 전체 CUDA workspace gate | `gpu-lock` 아래에서 `cargo test --locked --workspace --profile test-fast --features cuda -j2 --no-fail-fast -- --test-threads=1`로 실행하여 doc test를 포함한 163개 suite에서 12,247 passed, 0 failed, 389 ignored로 통과 |
| 전체 CUDA Clippy | `cargo clippy --locked --workspace --all-targets --profile test-fast --features cuda -j 2 -- -D warnings` 통과 |

PR #2270이 보존한 GB10 evidence에서 normalized state RMS/max는 tolerance 변경 없이 `2.532402801e-3 / 4.127457397e-2`에서 `2.838613609e-8 / 1.701009653e-6`으로 줄었습니다. Deterministic Granite 비교에서는 fused-kernel과 graph가 동일한 64-token sequence를 생성했습니다.

## 5. 리뷰 결과와 한계

구현 리뷰는 회귀 테스트가 독립적이며 기존 test flow에 올바르게 통합되었음을 확인했습니다. 보안 리뷰에서 새로운 input, memory, privilege, data-exposure surface는 발견되지 않았습니다. 성능 영향은 test 실행에만 있으며 production code는 현재 `main`과 같습니다.

Runtime 검증 범위는 GB10 CUDA입니다. Metal과 ROCm runtime 실행, FalconH1과 Plamo2의 실제 checkpoint 검증은 이 PR의 현재 범위 밖입니다.

## 6. 관련 기록

- Issue #2142: CUDA SSM BF16 parity failure
- PR #2209: 원래 진단 및 회귀 테스트 기여
- PR #2270: production 수정 및 보존된 GB10 수치/checkpoint evidence
- `docs/benchmark_results/data/ssm-bf16-parity-gb10-2026-10-11/`: stage deviation 및 Granite kernel-versus-graph artifact
