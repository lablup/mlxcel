# PR #2711: ROCm imagegen parity 테스트에 백엔드 반올림 차이 허용

**날짜**: 2026-10-28 (머지 전 작성, PR은 `5cbf1bb8`에서 열린 상태)
**상태**: ROCm gfx1151에서만 검증, Metal과 CUDA는 실행하지 않음
**언어**: Rust (테스트만)
**위험도**: 낮음 (프로덕션 코드 변경 없음. 위험은 비트 일치하지 않는 백엔드에서 느슨해진 경계가 향후 포팅 버그를 가릴 수 있다는 점)

## 요약

PR #2711 (#2709 해결)은 Ideogram 4 포팅(#2703)과 Qwen-Image 포팅(#2706)에서 추가된 imagegen parity 테스트 다섯 개가 ROCm gfx1151에서 통과하도록 합니다. 이 테스트들은 mlx 0.32.4 CUDA에서 기록한 fixture와 비교하면서, 비트 단위 일치나 CUDA를 비트 그대로 재현하는 백엔드에서만 성립하는 경계를 요구했습니다. #2709의 조사 결과 커널 결함도 포팅 버그도 없었습니다. 실패 원인은 초월 함수의 마지막 비트 차이와 bf16 반올림이며, fixture로 쓰는 작은 랜덤 모델이 이 차이를 증폭합니다.

변경은 테스트 코드에만 있습니다. 노이즈 샘플은 2 ulp 이내, RoPE 테이블은 절대 오차 1e-4 이내로 비교하고, bf16 forward 경계는 백엔드에 따라 달라집니다. Metal과 CUDA는 기존의 엄격한 1e-3 (Ideogram 4)과 2e-3 (Qwen-Image)을 유지하고, 그 밖의 백엔드(ROCm, CPU 전용 빌드)는 5e-2와 2e-2를 씁니다. 느슨해진 bf16 경계로는 ROCm에서 디바이스 커널 결함을 더 이상 잡을 수 없으므로, Ideogram 4 forward를 f32로 GPU와 CPU 스트림에서 각각 실행해 1e-4 이내로 일치하는지 확인하는 테스트를 새로 추가했습니다. ROCm에서 치르는 비용은 bf16 노이즈 바닥보다 작은 포팅 버그(측정된 예: fused RMSNorm 변형, 1.5e-2)를 Metal과 CUDA에서만 잡을 수 있다는 점입니다.

## 1. 문제 정의

gfx1151 호스트의 main에서 `make verify-rocm-held`가 다섯 테스트에서 실패했습니다.

| 테스트 | 검사 | ROCm 결과 |
|---|---|---|
| `ideogram4::pipeline::tests::noise_reproduces_upstream_global_seeding` | 정확한 `assert_eq!` | 원소 5가 1 ulp 차이 (-1.3240628 대 -1.3240626) |
| `ideogram4::transformer::tests::the_rope_tables_match_upstream` | cos, sin에 정확한 `assert_eq!` | 최대 차이 3.2e-6 (cos), 3.0e-5 (sin) |
| `ideogram4::transformer::tests::conditional_and_unconditional_forwards_match_upstream` | rel-RMS < 1e-3 | conditional 1.41e-2, unconditional 2.37e-2 |
| `qwen_image::transformer::tests::generate_forward_matches_upstream` | rel-RMS < 2e-3 | 5.9e-3 |
| `qwen_image::transformer::tests::edit_forward_with_and_without_the_kv_cache_matches_upstream` | rel-RMS < 2e-3 | 2.8e-3 |

허용 오차를 바꾸기 전에 #2709가 답해야 했던 질문은 이 수치가 ROCm overlay 커널 결함(mlxcelverse overlay와 `LOCAL_FIXES.md`에서 고칠 문제)에서 오는지, 포팅 버그(모델 코드에서 고칠 문제)에서 오는지, 반올림(테스트에서 고칠 문제)에서 오는지였습니다.

### 반올림이라고 판단한 근거

- **초월 함수.** 노이즈 샘플은 `erfinv`와 `log`를, RoPE 테이블은 `pow`, `sin`, `cos`를 거칩니다. 이 함수들은 CUDA, CPU 스트림, ROCm 사이에서 마지막 한두 비트가 다릅니다. CPU 스트림은 이미 노이즈 원소 510에서 CUDA와 1 ulp 차이가 났고, 기존 테스트는 꼬리 부분에만 절대 오차 1e-7을 허용해 이를 처리했습니다. RoPE 인자는 `position * inv_freq`이고 position이 65536을 넘으므로, `inv_freq`의 1 ulp 차이가 테이블에서 약 3e-5로 커집니다. 모델이 쓰기 전에 테이블은 bf16(1 근처 간격 약 4e-3)으로 반올림되므로 이 차이는 모델에 도달하지 않습니다.
- **혼돈적인 작은 모델.** fixture는 랜덤 bf16 가중치를 가진 작은 모델입니다. upstream 자체의 bf16 forward가 같은 가중치의 f32 forward와 0.37 (Ideogram 4), 2.9e-2 (Qwen-Image) 차이가 나므로, 이 네트워크는 학습된 모델보다 bf16 비트 하나의 차이를 훨씬 크게 증폭합니다. 비트 단위로 일치하지 않는 두 bf16 실행은 약 1e-2 떨어진 결과를 냅니다.
- **CPU 대조군.** CPU 스트림도 CUDA 기준과 어긋납니다 (Ideogram 4 conditional 1.61e-2, Qwen-Image 5.2e-3). 즉 bf16 차이는 ROCm 고유의 문제가 아닙니다. Metal과 CUDA가 이 크기에서 우연히 비트 단위로 일치할 뿐이고, 다른 백엔드는 그렇지 않습니다.
- **f32 대조군.** 같은 가중치의 f32 forward는 ROCm GPU와 CPU 스트림 사이에서 3e-5 (Ideogram 4), 7e-7 (Qwen-Image) 이내로 일치합니다. RMSNorm, attention, matmul, RoPE 적용, 마스킹 중 하나라도 결함이 있으면 여기서 드러났을 텐데, 드러나지 않았습니다.
- **발산 지점.** Ideogram 4에서 ROCm GPU와 CPU의 첫 발산은 timestep MLP에 있습니다. 들어가는 sinusoid는 동일하고 나오는 값이 1.7e-3 다르며, 이는 bf16 matmul과 bias 반올림입니다. 이후 모든 레이어가 이를 증폭합니다. Qwen-Image에서는 ROCm의 `t = 0.5` 케이스가 upstream과 정확히 일치하고 `t = 0.875` 케이스는 그렇지 않습니다. timestep 주파수와 sinusoid를 호스트에서 f64로 계산해도 결과가 바뀌지 않았으므로, timestep embedding의 초월 함수는 원인이 아닙니다.

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `src/imagegen/ideogram4/pipeline_tests.rs` | 새 헬퍼 `ulps(a, b)` (같은 부호 유한값의 비트 거리, 부호가 다르면 `u32::MAX`)와 `MAX_NOISE_ULPS = 2`로 게이트하는 `assert_noise`. 세 노이즈 구간(seed 0 앞부분, seed 0의 510 이후 꼬리, seed 7 앞부분)이 모두 이를 쓰며, 정확 비교 `assert_eq!` 두 개와 꼬리 전용 1e-7 허용을 대체합니다. |
| `src/imagegen/ideogram4/transformer_tests.rs` | RoPE 테이블 테스트는 원소별로 `MAX_TABLE_ABS_ERR = 1e-4` 이내인지 비교하고 테이블 이름과 인덱스를 보고합니다. `MAX_REL_RMS`는 `max_rel_rms()`가 되어 `gpu_backend_kind()`가 Metal 또는 CUDA면 1e-3, 그 외에는 5e-2를 반환하며, 측정값을 doc comment에 기록했습니다. forward 테스트는 이제 `lock_default_device()`를 잡습니다. 새 테스트 `the_f32_forward_agrees_between_the_gpu_and_the_cpu`. |
| `src/imagegen/qwen_image/transformer_tests.rs` | `TINY_REL_RMS`는 `tiny_rel_rms()`가 되어 Metal과 CUDA에서 2e-3, 그 외에는 2e-2이며, 측정값을 doc comment에 기록했습니다. |

프로덕션 코드, overlay, `LOCAL_FIXES.md` 항목은 바뀌지 않았습니다.

### 새 f32 테스트

`the_f32_forward_agrees_between_the_gpu_and_the_cpu`는 fixture의 모든 가중치(`w.*`)를 f32로 넓혀 `Ideogram4Transformer`를 두 번 만듭니다. 한 번은 기본 GPU 디바이스에서, 한 번은 `DefaultDeviceGuard::cpu()` 안에서 만들고, conditional과 unconditional forward를 모두 실행합니다. 두 경우 모두 rel-RMS가 1e-4 미만이어야 합니다 (gfx1151 측정값 3e-5, 5e-5). GPU 백엔드가 없으면 비교할 대상이 없으므로 바로 반환합니다. 프로세스 전역 기본 디바이스를 옮기므로 전체 구간 동안 `lock_default_device()`를 잡고, 같은 파일의 bf16 forward 테스트도 같은 락을 잡아 병렬 `cargo test`에서 둘이 섞이지 않게 합니다.

## 3. 기술적 선택과 그 이유

### 3.1 코드가 아니라 테스트를 고침

허용 오차를 늘리는 대신 ROCm이 CUDA를 비트 단위로 재현하게 만드는 방법도 있었습니다. matmul 누적 순서를 맞추고, 같은 `exp`, `erfinv`, `sin` 구현을 쓰고, timestep 경로를 호스트에서 계산하는 식입니다. 조사에서 가장 저렴한 단계(Qwen-Image의 timestep 주파수와 sinusoid를 호스트 f64로 계산)를 시도했지만 결과가 바뀌지 않았으므로, 발산은 일반적인 bf16 커널 전반에 퍼져 있습니다. 한 백엔드 전체를 다른 백엔드와 비트 호환으로 만드는 것은 overlay의 목표가 아니고, 아무도 고장났다고 보지 않는 CPU 스트림도 같은 폭으로 CUDA와 어긋납니다. fixture는 구조상(랜덤 가중치) 반올림에 민감하므로, 잘못 보정된 쪽은 테스트였습니다.

### 3.2 하나의 느슨한 경계 대신 백엔드별 경계

모든 백엔드에 5e-2 하나를 쓰는 편이 단순했겠지만, 그러면 테스트가 가진 가장 강한 증거를 버리게 됩니다. Metal과 CUDA에서는 포팅이 upstream을 비트 단위로 재현하고, 거기서 1e-3 경계는 pairwise (Z-Image) 회전, fused RMSNorm 커널, biased projection의 마스크 누락을 잡아냅니다. `gpu_backend_kind()`로 경계를 나누면 그 검사가 작동하는 곳에서는 유지되고, 다른 곳에서 작동하지 않는 이유가 문서로 남습니다. `GpuBackendKind::None` (CPU 전용 빌드)은 느슨한 쪽에 들어가며, 이는 측정된 CPU 스트림 차이 1.61e-2와 맞습니다.

### 3.3 5e-2와 2e-2의 선택

경계는 측정된 노이즈와 측정된 변형(mutation) 사이에 놓였습니다.

| 모델 | ROCm과 CPU의 노이즈 | 느슨한 경계 | 잡히는 변형 (ROCm) | 놓치는 변형 (ROCm) |
|---|---|---|---|---|
| Ideogram 4 | 1.41e-2 ~ 2.37e-2 | 5e-2 | pairwise 회전 5.2e-1, 3.2e-1, 마스크 누락 2.0e-1 | fused RMSNorm 1.5e-2 |
| Qwen-Image | 2.8e-3 ~ 5.9e-3 | 2e-2 | 측정 안 함 | 측정 안 함 |

Ideogram 4는 측정된 최악의 노이즈 대비 약 2배 여유가 있고, 가장 가까운 잡히는 변형은 경계의 4배 위에 있습니다. Qwen-Image는 노이즈 대비 약 3.4배 여유가 있지만 느슨한 경계에 대한 변형을 측정하지 않았으므로, ROCm에서의 검출력은 알 수 없습니다.

### 3.4 잃어버린 검사를 대신하는 f32 디바이스 간 테스트

ROCm에서 bf16 경계를 느슨하게 하면 반올림 차이와 고장난 디바이스 커널을 구분할 수 없게 됩니다. f32 forward는 bf16 증폭 대부분을 제거하므로 (GPU와 CPU 차이가 약 1e-2 대신 3e-5), 여기서 1e-4 경계는 디바이스 경로의 잘못된 norm, attention, matmul, 마스크를 잡을 만큼 엄격합니다. 저장된 f32 fixture가 아니라 같은 프로세스 안에서 GPU와 CPU를 비교하면 새 fixture가 필요 없고, overlay가 책임지는 부분인 디바이스 커널만 정확히 분리됩니다.

이 테스트가 검사하는 범위는 bf16 parity 테스트보다 좁습니다. 포팅 버그는 두 디바이스에서 똑같이 계산되므로, GPU와 CPU가 일치한다는 것은 포팅이 upstream과 맞는지에 대해 아무것도 말해 주지 않습니다. ROCm에서 5e-2 미만의 포팅 충실도는 이제 Metal과 CUDA의 엄격한 경계 실행으로만 보장됩니다.

### 3.5 노이즈는 ulp, 테이블은 절대 오차

노이즈 값은 여러 자릿수(1e-2부터 1.4까지)에 걸쳐 있으므로, 하나의 절대 오차는 작은 값에는 너무 느슨하거나 큰 값에는 너무 엄격합니다. 2 ulp 게이트는 값의 크기에 따라 조정되면서도, 모든 원소를 완전히 바꾸는 잘못된 키 유도나 샘플링 순서는 여전히 거부합니다. RoPE 테이블 원소는 [-1, 1] 범위의 cos, sin 값이고 의미 있는 척도는 반올림되는 bf16 간격이므로, 절대 오차 1e-4 경계(그 간격보다 40배 작고, 측정값 3.0e-5보다 3배 큼)가 요구 사항을 직접 표현합니다.

## 4. 검증

- gfx1151에서 `cargo test --lib --profile test-fast --features rocm imagegen:: -- --test-threads=1`: 374 통과, 0 실패, 30 무시. 변경 전에는 1절의 다섯 테스트가 실패했습니다.
- `cargo clippy --lib --tests --features rocm -- -D warnings`, `cargo fmt --check` 통과.
- 실행 안 함: Metal과 CUDA. 이 백엔드들은 bf16 forward에 기존의 엄격한 경계를 유지하지만, 노이즈와 RoPE 테스트는 그곳에서도 느슨해졌고, 새 f32 테스트는 그곳에서 처음 실행됩니다.

## 5. 학습 포인트

- **작은 랜덤 모델은 반올림 증폭기입니다.** 같은 랜덤 가중치의 bf16과 f32 forward가 0.37 차이 난다면, 비트 일치하지 않는 bf16 백엔드는 모두 약 1e-2 어긋납니다. 비트 일치 백엔드에서 보정한 parity 경계는 다른 백엔드가 달성할 수 있는 수준에 대해 아무것도 말해 주지 않습니다. 백엔드 간 경계를 정하기 전에 fixture의 bf16 대 f32 차이를 측정해야 합니다.
- **CPU 스트림을 대조군으로 쓰십시오.** CPU 스트림이 GPU와 같은 폭으로 기준과 어긋난다면 GPU가 문제가 아닙니다. 이 검사는 저렴하고, #2709의 질문 대부분을 해결했습니다.
- **커널 정확성과 포팅 충실도를 분리하십시오.** GPU와 CPU의 f32 실행은 디바이스 커널을 검사하고, upstream fixture와의 비교는 포팅을 검사합니다. 하나의 느슨한 테스트로는 둘 다 할 수 없으며, 이 PR은 어느 백엔드가 어느 속성을 보장하는지 명시합니다.
- **주기 함수에 큰 인자를 넣으면 ulp가 증폭됩니다.** position이 65536을 넘는 `sin(position * inv_freq)`는 `inv_freq`의 1 ulp 변화를 출력에서 3e-5로 만듭니다. RoPE 테이블의 정확한 일치는 이식성이 없습니다.
- **기본 디바이스를 옮기는 테스트는 전역 락을 잡아야 합니다.** `DefaultDeviceGuard`는 디바이스를 복원할 뿐 다른 스레드를 배제하지 않고, 배제는 `lock_default_device()`(재진입 불가)가 합니다. 같은 파일에서 기본 디바이스에 의존하는 테스트도 이 락을 잡아야 합니다.

## 6. 후속 과제와 한계

- **Metal과 CUDA에서 f32 경계는 측정되지 않았습니다.** `the_f32_forward_agrees_between_the_gpu_and_the_cpu`의 1e-4 경계는 ROCm에서만 측정했습니다. f32 matmul을 낮은 정밀도로 계산하는 백엔드(예: CUDA에서 TF32가 켜져 있는 경우)는 이를 넘을 수 있습니다. Metal이나 CUDA의 첫 실행에서 여유를 확인해야 합니다.
- **Qwen-Image에는 f32 디바이스 간 테스트가 없습니다.** ROCm에서는 bf16 경계가 유일한 검사이고, 2e-2에서의 변형 민감도는 측정하지 않았습니다. 조사 중 f32 대조값(7e-7)은 측정했지만 테스트로 커밋하지는 않았습니다.
- **비트 일치하지 않는 백엔드에서 노이즈보다 작은 포팅 버그.** fused RMSNorm 커널(1.5e-2) 같은 변경은 ROCm과 CPU 전용 빌드에서 통과합니다. 이는 Metal이나 CUDA가 정기적으로 테스트를 실행하는 동안에만 허용 가능하며, 호스팅 CI가 중단된 동안의 ROCm 전용 검증은 이를 잡지 못합니다.
- **테스트 모음의 다른 parity 테스트.** 실패한 다섯 테스트만 재보정했습니다. CUDA에서 기록하고 정확 일치나 비트 일치 기준 경계를 쓰는 다른 fixture도 새 백엔드에서 같은 방식으로 실패할 수 있으며, 여기서 쓴 패턴(백엔드별 경계와 f32 디바이스 간 대조 테스트)이 그 템플릿입니다.
- **기본 디바이스 락 적용 범위.** Qwen-Image 테스트는 `lock_default_device()`를 잡지 않습니다. `--test-threads=1`에서는 문제가 없지만, 병렬 `cargo test --lib`에서는 다른 모듈이 CPU 기본 디바이스를 잡고 있는 동안 실행되어 엉뚱한 디바이스에서 측정될 수 있습니다.

## 참고

- PR #2711: https://github.com/lablup/mlxcel/pull/2711
- Issue #2709: ideogram4와 qwen_image의 ROCm parity 테스트가 CUDA 비트 일치를 요구
- PR #2703: Ideogram 4 text-to-image 포팅 (Ideogram 4 fixture의 출처)
- PR #2706: Qwen-Image-2.1 생성과 편집 포팅 (Qwen-Image fixture의 출처)
- Issue #1421: 프로세스 전역 기본 디바이스 락 (`lock_default_device`)
- `src/lib/mlxcel-core/src/streams.rs` (`DefaultDeviceGuard`, `lock_default_device`), `src/lib/mlxcel-core/src/hardware.rs` (`GpuBackendKind`)
