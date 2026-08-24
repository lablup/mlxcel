# 기술 보고서: PR #1401 - Shapeless MLX compile 지점 감사

**작성일**: 2026-08-25
**상태**: 완료
**언어**: Rust, C++, Bash
**위험도**: Medium

## 요약

PR #1401은 `mlx::core::compile(..., shapeless=true)`에 대한 일회성 수동 조사를 반복 가능한 하드웨어 감사로 전환한다. `mlx_cxx_bridge.cpp`의 모든 production shapeless compile 생성 지점은 opt-in eager oracle을 통과하며, 조용한 no-op가 그럴듯한 텐서를 반환할 수 있는 경로는 영구 회귀 테스트로 보호한다.

최종 구현은 모든 compiled fusion을 유지한다. 로컬 NVIDIA CUDA와 Apple Silicon Metal에서 현재 20개 callable을 f32, bf16, f16으로 가능한 범위까지 검사했고, 최초 호출과 warm-cache 재호출이 모두 통과했다. 모델 체크포인트는 읽거나 다운로드하지 않았다.

## 1. 문제 정의

PR #1391은 Metal에서 입력을 그대로 반환하던 compiled min-p filter를 제거했다. 같은 C++ bridge에는 여러 shapeless compile 생성 지점이 남아 있었고, 출력 shape만으로는 softcap이나 residual clipping처럼 입출력 shape가 같은 연산을 안전하다고 판단할 수 없었다.

구현 시점에는 이슈의 원래 line/function 표도 실제 코드와 달라져 있었다. `softplus`는 이미 eager였고, QKV로 적힌 위치는 quantized MoE expert 경로였으며, masked/unmasked softcap SDPA는 서로 다른 두 compiled callable이었다. 현재 실제 inventory는 20개다.

감사 없이 MLX pin을 올리면 activation, attention transform, state update가 조용히 비활성화되어도 shape와 dtype이 정상인 텐서를 반환할 수 있다. 특히 filter나 bounded transform은 no-op 결과도 수치적으로 그럴듯해 위험하다.

## 2. 기술적 선택과 그 이유

### 2.1 중앙 opt-in eager oracle

모든 shapeless 생성 지점은 `compile_shapeless_audited(site, eager_fn)`을 호출한다. `MLXCEL_SHAPELESS_COMPILE_AUDIT`가 설정되지 않으면 기존 compiled callable을 그대로 반환하며 호출마다 비교하거나 registry를 갱신하지 않는다. 첫 compiled function 초기화 전에 audit를 활성화하면 동일 입력으로 compiled/eager graph를 실행해 output count, shape, dtype, numeric closeness를 비교하고 dtype/shape signature별 호출 수를 기록한다.

이 방식은 20개 production callable에 대해 별도 handwritten reference graph를 중복 유지하지 않는다. MLX compile에 넘긴 정확히 같은 lambda가 eager oracle이므로, 나중에 한쪽 구현만 바뀌는 drift도 방지한다.

### 2.2 실측 후 fusion 보존

중간 구현은 softcap, clip-residual, softcap SDPA에서 compile을 선제적으로 제거했다. 하지만 실제 divergence 근거 없이 attention fusion을 버리면 성능 회귀가 될 수 있어 이 접근을 폐기했다. 최종 CUDA/Metal 감사에서 해당 지점이 모두 통과했으므로 production 최적화는 유지하고, 별도의 독립 reference 회귀 테스트로 의미론을 고정했다.

### 2.3 모델 CI 확대가 아닌 하드웨어 감사

`scripts/audit_shapeless_compile.sh`는 작은 synthetic tensor만 사용한다. Linux에서는 `nvidia-smi`가 보고하는 연결 GPU의 compute capability를 읽어 `MLX_CUDA_ARCHITECTURES`를 설정하고, macOS에서는 `metal,accelerate`를 선택한다. 중앙 wrapper를 우회하는 direct shapeless compile을 거부하며, 최소 Apple runner에 `rg`가 없으면 `grep`으로 inventory를 검사한다.

영구 GitHub Actions job은 추가하지 않았다. Apple 감사용 branch-only workflow는 PR 생성 전에 제거되어 Actions diff가 없다. 기존 프로젝트 경계도 유지한다. Hosted Actions는 약 0.6B 수준의 기존 fixture를 사용할 수 있지만, 더 큰 체크포인트 검증은 로컬 하드웨어와 이미 존재하는 weights로 수행한다.

## 3. 구현 상세

### 3.1 감사 흐름

```text
Production (environment unset)
call site -> compile_shapeless_audited -> original compiled callable

On-demand audit (MLXCEL_SHAPELESS_COMPILE_AUDIT=1)
call site -> compiled graph ----+
             eager graph -------+-> shape/dtype/allclose -> per-site report
                                      first call + warmed call per signature
```

### 3.2 영구 quiet-site 회귀 테스트

- `compiled_softcap`: cap보다 큰 양수/음수 nonuniform logits, 명시적 `tanh(scores / cap) * cap` reference, 결과가 입력과 다르다는 assertion.
- `compiled_clip_residual`: f16 overflow boundary 값, 명시적 f32 widen/add/clip/f16 reference, 결과가 첫 입력과 다르다는 assertion.
- Masked/unmasked softcap SDPA: nonuniform Q/K/V와 독립 eager attention reference를 f32, bf16, f16으로 비교.
- GQA softcap SDPA: explicit repeated-K/V eager attention reference로 shape-only 테스트가 그럴듯한 no-op를 허용하지 않도록 고정.

## 4. 하드웨어 감사 기준선

CUDA host는 NVIDIA GB10, driver 580.173.02, CUDA 13 runtime 가시성, compute capability 12.1 환경이다. Metal 결과는 `self-hosted-macos-26-arm64`에서 실행한 GitHub Actions [32746852051](https://github.com/lablup/mlxcel/actions/runs/32746852051)이다.

`6 / 3 / 3`은 6회 호출, 3개 dtype/shape signature, 3개 signature 모두 2회차 warm 호출 완료를 뜻한다. Clip-residual은 의도적으로 f16 전용이므로 `2 / 1 / 1`이다.

| Site | Coverage (calls / signatures / warmed) | CUDA | Metal |
|---|---:|---|---|
| `compiled_swiglu_activation` | 6 / 3 / 3 | PASS | PASS |
| `compiled_relu_squared` | 6 / 3 / 3 | PASS | PASS |
| `compiled_silu` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gpt_oss_swiglu_activation` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_approx` | 6 / 3 / 3 | PASS | PASS |
| `compiled_geglu_activation` | 6 / 3 / 3 | PASS | PASS |
| `compiled_geglu_approx_activation` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_topk` | 6 / 3 / 3 | PASS | PASS |
| `compiled_softcap` | 6 / 3 / 3 | PASS | PASS |
| `compiled_clip_residual` | 2 / 1 / 1 | PASS | PASS |
| `compiled_softcap_sdpa_nomask` | 6 / 3 / 3 | PASS | PASS |
| `compiled_softcap_sdpa_masked` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_mlp_forward` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_approx_mlp_forward` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_approx_mlp_forward_global_scale` | 6 / 3 / 3 | PASS | PASS |
| `compiled_per_layer_input_gate` | 6 / 3 / 3 | PASS | PASS |
| `compiled_moe_expert_forward` | 6 / 3 / 3 | PASS | PASS |
| `fused_gated_delta_decode_step_scalar_gate` | 6 / 3 / 3 | PASS | PASS |
| `fused_gated_delta_decode_step_dim_gate` | 6 / 3 / 3 | PASS | PASS |

## 5. 호환성, 보안, 성능

- **Breaking changes**: 없음. 기존 bridge function 이름과 production compiled 동작은 유지된다.
- **새 의존성**: 없음. 감사 스크립트는 선택 backend의 기존 도구만 사용하고 `rg`/`grep` inventory fallback을 제공한다.
- **보안**: 감사 기능은 opt-in이며 synthetic data만 사용한다. Prompt, token, credential, model weight를 읽지 않고 report에는 site 이름과 tensor signature만 남긴다.
- **성능**: Audit-disabled 호출은 원래 compiled function을 반환한다. Quiet-site fusion은 CUDA/Metal 직접 검증 후 유지했다.
- **PR #1395**: Branch base에 이미 포함되어 있지만 MLA KV-cache/Turbo4 안전성 수정으로, MLX compile 동작과는 무관하다.

## 6. 변경 요약

이 보고서 추가 전 implementation diff:

| 항목 | 값 |
|---|---:|
| 변경 파일 | 5 |
| 추가 라인 | 843 |
| 삭제 라인 | 114 |
| 영구 회귀 테스트 | 4 |
| 감사한 shapeless callable | 20 |
| 새 runtime 의존성 | 0 |
| 영구 workflow 변경 | 0 |

주요 파일:

- `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`: 중앙 wrapper, registry, report, 모든 production site routing.
- `src/lib/mlxcel-core/src/ffi_tests.rs`: multi-dtype hardware harness와 quiet-site 회귀 테스트.
- `scripts/audit_shapeless_compile.sh`: backend 선택, source inventory 강제, isolated ignored test 실행.
- `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.h`, `src/lib/mlxcel-core/src/lib.rs`: audit report bridge.

## 7. 검증

- 로컬 CUDA `scripts/audit_shapeless_compile.sh`: 20/20 PASS, 19개 site `6/3/3`, clip-residual `2/1/1`.
- Apple Silicon Metal run 32746852051: 동일한 20/20 표, quiet-site 회귀 4/4 PASS.
- `cargo test -p mlxcel-core --profile test-fast --features cuda --lib eager_regression -- --nocapture --test-threads=1`: 4 passed.
- `cargo test -p mlxcel-core --profile test-fast --features cuda --lib compiled_ -- --test-threads=1`: 30 passed.
- `cargo clippy -p mlxcel-core --features cuda --all-targets -- -D warnings`: 통과.
- `cargo fmt --all -- --check`: 통과.

## 8. 후속 조치

- MLX pin을 올릴 때마다 CUDA와 Apple Silicon에서 `scripts/audit_shapeless_compile.sh`를 재실행한다.
- 누락 site, warm되지 않은 signature, `FAIL` 행은 qualification 실패로 처리하고, 실측 divergence가 있는 site만 compile을 제거하거나 정확한 safe precondition을 문서화한다.
- Real-checkpoint 검증은 별도로 유지하며, hosted fixture 경계보다 큰 체크포인트는 로컬의 기존 weights를 사용한다.

## 참고 자료

- Issue #1392: shapeless compile 감사 요청.
- PR #1391: 동기를 제공한 compiled min-p no-op 수정.
- PR #1395: base에 포함된 인접 변경이지만 MLA KV-cache/Turbo4 안전성 수정으로 본 감사와 무관.
