# 기술 보고서: PR #2035 - Affine 1-bit 체크포인트를 fused Metal 커널로 실행

**날짜**: 2026-09-29

**상태**: Apple M1 Ultra(Metal)에서 구현 및 검증 완료, 머지 대기.

**언어**: C++ (Metal 커널, 브리지 라우팅), Rust (로더 검증, 레이어 라우팅, 테스트)

**위험도**: 중간

## 요약

MLX는 2, 3, 4, 5, 6, 8비트 affine 양자화 커널만 제공한다. 공개된 1-bit 체크포인트(`prism-ml/Bonsai-{1.7B,4B,8B,27B}-mlx-1bit`, 일반 `qwen3`)는 표준 MLX 패킹을 1비트로 사용하며, mlxcel의 모든 로드 검사를 통과한 뒤 첫 토큰에서 프로세스를 중단시켰다. 이 PR은 mlxcel 자체 1-bit matvec, matmul 커널을 추가하고, 브리지 수준의 모든 양자화 프리미티브를 이 커널로 라우팅하며, MLX를 직접 호출하는 fused C++ 헬퍼가 1-bit 가중치를 받지 않도록 한다.

## 1. 문제

main에서 `mlxcel generate -m Bonsai-1.7B-mlx-1bit`는 0.12초 만에 로드된 뒤 `libc++abi: terminating due to uncaught exception of type std::runtime_error: [metal::Device] Unable to load kernel affine_dequantize_float16_t_gs_128_b_1`(exit 134)를 출력했다. 첫 실패 지점은 matmul이 아니라 임베딩 조회의 dequantize였다. `validate_quantization_params`는 범위 검사(1..=32)이고, `infer_quantization_bits`는 shape가 일치하면 선언값을 바로 반환하므로 {2..8} 허용 목록이 참조되지 않아 로드가 성공했다.

## 2. 레이아웃과 수식

`N`행 `K`열 linear에서 `weight`는 `uint32 [N, K/32]`이고 word `c`의 비트 `j`(LSB 우선)가 열 `32c + j`이다. `scales`, `biases`는 `[N, K/G]`, `G`는 {32, 64, 128}. 역양자화 값은 `w = bit * scale + bias`. bias는 비트와 무관하므로 matvec은 (행, 그룹)마다 두 개의 합만 필요하다. 비트 마스크를 적용한 활성값 합과 전체 활성값 합. `y = sum_g scale * masked_sum + bias * total_sum`.

## 3. 변경 사항

- `cpp/mlx_cxx_one_bit.cpp`(신규): qmv(64스레드 threadgroup, R = 4 또는 8행을 맡는 simdgroup 2개, lane당 한 스텝 16열, `simd_sum` 축약, 행 경계 검사를 없애는 `ALIGNED` 템플릿), qmm(32x32 타일, 128스레드, 비트를 threadgroup 메모리에서 `bias` 또는 `scale + bias`로 전개, 8x8 simdgroup fragment, f32 누적, 프롬프트 길이마다 커널이 새로 컴파일되지 않도록 `m_size`를 스칼라 입력으로 전달), `uint8` view로 언패킹하는 dequantize 그래프(`uint32` shift 대비 임시 메모리 1/4).
- `mlx_cxx_bridge.cpp`: `quantized_matmul`(두 `transpose` 레이아웃), `quantized_linear_forward`, `quantized_linear_forward_global_scale`, `dequantize`, `quantized_embedding`이 biases가 있는 affine `bits == 1` 트리플을 새 파일로 보낸다.
- `layers.rs`: `QuantizedWeight::is_one_bit`, fused 헬퍼에 가중치를 넘기는 두 `UnifiedLinear` 접근자가 1-bit에 `None` 반환, su-scaled-rope 경로와 compiled `SwiGLUMLP`의 명시적 거부, prefill이 qmm을 쓰도록 1-bit에서 #1994 dense-GEMM prefill 생략, `LOADABLE_AFFINE_BITS`와 `validate_one_bit_layout`, `QuantizedMultiLinear`의 1-bit 거부.
- `switch_layers.rs`: 1-bit expert stack을 prefix와 함께 로드 시점에 거부.
- FFI: `one_bit_quantized_matmul`, `one_bit_dequantize`(둘 다 `Result`), `one_bit_kernel_available`.

## 4. 기술적 결정

### 라우팅은 브리지에서, 거부는 접근자에서

양자화 트리플로 MLX에 도달하는 호출자는 두 종류다. 브리지 프리미티브(`ffi::quantized_matmul` 등)는 `UnifiedLinear`, `QuantizedEmbedding`, 약 30개 모델 파일에서 호출되며, 그 내부에서 라우팅하면 이슈에 없던 호출자까지 한 번에 처리된다. fused C++ 헬퍼는 `mlx::core::quantized_matmul`을 직접 호출하므로 하나씩 고치지 않고는 라우팅할 수 없다. 이런 헬퍼는 모두 `quantized_weight()`, `as_quantized_weight()`, `fused_quantized_weight()`로 게이트되어 있고, runtime LoRA(#1439)가 이미 `None`을 "그래프 fallback 사용"으로 정의해 두었다. 1-bit에서 `None`을 반환하면 이 계약을 그대로 재사용하므로 새 fallback 코드가 필요 없다.

### 허용 목록을 둘로 유지

`SUPPORTED_AFFINE_BITS`는 MLX affine quantize가 만들 수 있는 폭이며 생산자(`split-mtp`, gemma4 모듈별 override)가 읽는다. 여기에 1을 추가하면 아무도 양자화할 수 없는 폭을 생산하게 된다. 로더는 새 `LOADABLE_AFFINE_BITS`를 읽는다.

### 오라클

kill switch는 `scale + bias`를 f32로 만든 뒤 배포 dtype으로 한 번 반올림한 가중치로 `x @ dequant(W)^T`를 계산한다. 단위 테스트는 커널을 이 경로와 비교하지 않고, 두 경로 모두를 디바이스 값 그대로의 호스트 f64 계산과 비교하므로 공통 버그가 숨을 수 없다.

### CUDA 보류

이슈는 CUDA qmv 포트 또는 보류 사유 기록을 요구했다. CUDA 호스트가 없었다(GB10 러너 중단). 한 번도 실행되지 않은 JIT 커널을 기본 활성화하면 잘못된 출력을 조용히 낼 수 있으므로 포트 테이블은 Metal 전용이며, CUDA/ROCm은 거부가 아니라 `has_kernel_port`를 통해 dequantize 그래프를 탄다.

## 5. 검증

- main에서의 Step 0: 1절의 중단 메시지.
- `one_bit_tests`(17개): 커널과 fallback 각각이 f64 오라클 대비 `max|y|`의 5e-3(f16/f32), 2e-2(bf16) 이내. M 1~40, K 256~4096, N 64~1003(비정렬 포함), G 32/64/128. dequantize는 f32/f16/bf16에서 호스트 수식과 정확히 일치, kill switch는 그래프와 비트 단위 일치, `transpose=false`, 임베딩 gather와 tied `as_linear`, fused QKV 거부, 로드 거부 메시지의 prefix 포함.
- `Bonsai-1.7B-mlx-1bit` greedy 64토큰: 커널과 fallback 출력 동일, 답 "Paris". `Bonsai-8B-mlx-1bit` greedy 128토큰: 동일, 자연스러운 영어. 1.7B에서 `mlxcel-server`로 동시 greedy chat 요청 2건: 모두 정답.
- 계약 테스트, `layers::tests`, `switch_layers::tests`, fmt, clippy(`-p mlxcel-core`, `-p mlxcel`) 통과.

## 6. 보류 항목

- `mlx-community/Qwen3-8B-4bit` 대비 디코드 처리량: 실행 중 GPU를 다른 작업과 공유했으므로 시간 수치는 공개하지 않는다.
- CUDA qmv 포트와 GB10 검증.
- 1-bit MoE expert stack과 MLA 투영(로드 시 거부, 사용하는 공개 체크포인트 없음).
