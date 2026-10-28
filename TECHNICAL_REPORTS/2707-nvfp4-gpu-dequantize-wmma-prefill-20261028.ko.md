# PR #2707: NVFP4 dequantize의 GPU 실행과 WMMA expert prefill 추가 (ROCm)

**날짜**: 2026-10-28 (머지 전 작성, PR은 `3cd30dbf`에서 open 상태)
**상태**: gfx1151에서만 검증, wave64 및 multi-arch fat 빌드는 빌드하지 않음
**언어**: HIP/C++ (mlxcelverse ROCm overlay), Rust (모델 주석, 테스트), Markdown
**위험도**: 중간 (MoE prefill의 기본 활성 경로에서 수치가 바뀜, 아무도 빌드하지 않은 아키텍처용으로 컴파일되는 새 device 코드)

## 요약

PR #2707 (#2698 종료)은 gfx1151에서 `mlx-community/Nemotron-Cascade-2-30B-A3B-nvfp4`의 NVFP4 prefill 처리량을 약 3배로, decode를 약 3분의 1 높입니다. 두 가지 변경이 효과를 냅니다. 첫째, GPU stream의 NVFP4 `dequantize`가 이제 CPU stream으로 넘어가지 않고 GPU에서 실행됩니다. 이로써 512 토큰 prefill의 약 3분의 2(임베딩 조회)와 매 decode step의 수 ms를 차지하던 host gap이 사라졌습니다. 둘째, expert-batched NVFP4 gather의 rocWMMA 타일 변형(`nvfp4_gather_expert_wmma_kernel`)이 expert당 평균 16행 이상일 때 정렬된 MoE prefill을 맡습니다.

512 토큰 prefill은 329.3에서 1007.5 tok/s로, 2048 토큰 prefill은 330.9에서 1105.7 tok/s로, decode는 45.2에서 60.0 tok/s(512 토큰 프롬프트)로 올랐습니다(같은 바이너리 3회 측정의 중앙값). 세 번째 후보였던 fused Mamba2 및 fused MoE decode 호출의 NVFP4 지원은 구현 후 측정했으나 빨라지지 않아 반영하지 않았고, 측정 결과를 코드 위치에 남겨 같은 시도를 근거 없이 반복하지 않도록 했습니다. WMMA 커널의 wave64 장치 동작과 multi-architecture 빌드는 검증되지 않았습니다.

## 1. 문제 정의

이슈 #2698은 #2665의 후속입니다. #2665는 Nemotron-H가 NVFP4 export를 로드하게 하고 정렬된 MoE prefill용 `nvfp4_gather_expert_batched_kernel`을 추가했습니다. 그 이후에도 512 토큰 prefill의 커널 트레이스에서 GPU가 wall time의 약 절반 동안 놀고 있었고, #2665의 LOCAL_FIXES item 69는 이 gap을 남겨 두었습니다. 이슈는 또한 affine 가중치만 받는 Nemotron-H의 fused decode 호출(`fused_mamba2_forward`, `fused_moe_forward`)에 NVFP4를 지원할지도 물었습니다.

트레이스로 위치를 좁혔습니다. 1.5 s prefill 중 약 1 s(트레이서 아래에서는 2.1 s)가 임베딩 row gather와 첫 레이어 사이에 있었고, 메인 스레드에는 host API 호출이 없었습니다. 원인은 overlay의 `fp_dequantize`(`mlx/ops.cpp`)였습니다. ROCm에서는 범용 `fp_dequantize` 커널(`quantized/fp_quantize.hip`)이 block-float scale을 디코드하지 못하므로, GPU stream의 모든 block-float `dequantize`를 CPU stream으로 돌렸습니다. `quantized_embedding`은 row gather 뒤에 `dequantize`를 하므로, NVFP4 임베딩 조회는 매번 CPU에서 fine-grained(캐시되지 않는 device-coherent) 메모리를 읽으며 토큰당 약 2 ms를 썼습니다. decode도 토큰마다 같은 비용, step당 약 4 ms의 GPU idle을 치렀습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `mlx/ops.cpp` (`fp_dequantize`) | GPU stream에서 global scale이 없고 출력이 f32, f16, bf16인 NVFP4는 CPU 경로 대신 `fast::Quantize` primitive(dequantize 방향)를 만듭니다. `MLX_ROCM_NVFP4_GPU_DEQUANTIZE=0`(정확히 `0`)이면 CPU 경로로 돌아갑니다. MXFP4, MXFP8, global-scale NVFP4, 그 밖의 출력 타입은 CPU 경로를 유지합니다. |
| `quantized/quantized.cpp`, `quantized.h` | `fast::Quantize::eval_gpu`가 `QuantizationMode::Nvfp4`를 `quantized.h`에 선언한 새 `nvfp4_dequantize`로 보냅니다. |
| `quantized/qmm.hip` (`nvfp4_dequantize`) | LOCAL_FIXES item 63이 GEMM 경로용으로 추가한 기존의 정확한 `nvfp4_dequantize_dense`를 감싼 검사 래퍼(`check_nvfp4_planes`, encoder 입출력 등록)입니다. 새 dequantize 커널은 없습니다. |
| `quantized/qmm.hip` (`nvfp4_gather_expert_wmma_kernel`) | 128 스레드 새 커널, `__half`와 `hip_bfloat16`으로 인스턴스화하며 warp당 16행 타일 2개를 씁니다. 라우팅은 #2665와 같습니다: block z는 expert z, `find_sorted_expert_run`이 run을 찾고, block z == E는 E 이상으로 라우팅된 행을 0으로 채웁니다. 64폭 K step마다 activation 32행을 block당 한 번 shared memory로 모으고, 32 lane warp 4개가 각각 가중치 타일의 32열 slab을 activation dtype으로 (정확하게) dequantize한 뒤 2x2개의 16x16x16 rocWMMA 타일을 f32로 누적합니다. f32 출력 타일은 A/B shared buffer와 메모리를 공유하고, global scale은 store 시 한 번 곱합니다. |
| `GatherQMM::eval_gpu` dispatch | #2665의 NVFP4 expert-batched 경로 안에서 다음을 모두 만족하면 WMMA 변형을 씁니다: f16 또는 bf16 activation, `K % 64 == 0`, `B / E >= MLX_ROCM_NVFP4_GATHER_WMMA_MIN_ROWS`(16), activation과 weight base 16바이트 정렬 및 scale 4바이트 정렬, `d.has_native_wmma()`, `wmma_qmm_wavefront_ok()`(wave32). `MLX_ROCM_NVFP4_GATHER_WMMA=0`이면 스칼라 커널을 유지합니다. grid x는 평균 run 타일 수의 2배이고 전체 타일 수로 상한을 둡니다. |
| `src/models/nemotron_h.rs` | `is_affine_quantized`의 주석만 변경: 기각된 fused NVFP4 decode 실험과 수치를 기록합니다. 동작 변화는 없습니다. |
| 문서 | `docs/environment-variables.md`(새 변수 3개), `docs/installation.md`(NVFP4 행: 새 수치, WMMA 출력 차이, 기각된 fused decode), overlay `LOCAL_FIXES.md` item 70. |
| 테스트 | `tests/rocm_nvfp4_quant.rs`: GPU `dequantize`와 임베딩 조회를 CPU `dequantize`와 비트 단위로 비교합니다. `tests/rocm_nvfp4_quant/gather.rs`: expert 16개, 512행에서 여러 타일 run, 단일 행 run, 빈 run, 마지막 expert를 넘는 행, global scale 유무, 부분 128열 block, f32(스칼라), f16, bf16으로 WMMA 경로를 돌리고, 변형을 끈 child process 재실행을 추가합니다. |

통계: 10개 파일, +616 / -8. 커밋: `7e7378b7`(구현), `3cd30dbf`(리뷰 수정: WMMA 경로의 정렬 조건, GPU `dequantize` 경로의 출력 타입 조건).

## 3. 기술적 선택과 그 이유

### 3.1 GEMM 경로의 정확한 dequantize를 `fast::Quantize` 뒤에서 재사용

범용 `fp_dequantize` 커널에 E4M3 scale 디코드를 넣거나 새 primitive를 추가하는 방법도 있었습니다. 대신 그래프가 기존 `fast::Quantize` primitive를 dequantize 모드로 만들고, 그 `eval_gpu`가 `nvfp4_dequantize_dense`를 호출합니다. 이 함수는 item 63이 이미 dequantize 후 GEMM 경로에서 쓰고 있고 정확합니다(E2M1과 E4M3의 곱은 유효 비트가 6개 이하이며 fp16 normal 범위 안에 있습니다). 그래서 host 코드 변경은 십여 줄에 그쳤고 CPU 경로와 비트 단위로 같은 결과가 나옵니다. GPU `dequantize`만 켰을 때 두 테스트 프롬프트의 greedy 출력이 바이트 단위로 같은 이유입니다.

경로는 의도적으로 좁습니다. `nvfp4_dequantize_dense`에 global scale 입력이 없으므로 global-scale NVFP4는 제외했고, MXFP4와 MXFP8은 여전히 scale을 아는 커널이 필요하며, f32/f16/bf16 외의 출력은 예외를 던지지 않고 CPU 경로를 유지합니다(`3cd30dbf`의 리뷰 수정). kill switch는 값을 정확히 `"0"`과 비교하는데, WMMA 스위치는 `parse_warp_kernel_env`를 씁니다. 문서에 각 규칙이 적혀 있습니다.

### 3.2 expert당 행 수로 게이트한 WMMA 변형

이 변형은 64 x 128 가중치 타일 하나를 shared memory에 dequantize하는 비용을 최대 32행에 나눠 씁니다. expert당 행이 적으면 block마다 하는 dequantize가 스칼라 커널의 FMA보다 비쌉니다. 대상 체크포인트(expert 128개, top-6)에서 변형 유무에 따른 prefill:

| 프롬프트 토큰 | expert당 행 | WMMA | 스칼라 |
|---|---|---|---|
| 64 | 3 | 319 | 389 |
| 128 | 6 | 526 | 615 |
| 256 | 12 | 765 | 768 |
| 512 | 24 | 1017 | 877 |

교차점이 12와 24 사이이므로 기본 임계값은 평균 16행(`B / E`, 정수 나눗셈)이고 `MLX_ROCM_NVFP4_GATHER_WMMA_MIN_ROWS`로 조정할 수 있습니다. 평균은 근사치입니다. 라우팅이 치우치면 일부 expert의 run이 타일 하나보다 짧을 수 있는데, 커널은 이를 올바르게 처리하지만(마스킹된 행은 run의 첫 행을 읽고 저장하지 않음) 최적화하지는 않습니다.

타일 형태도 측정했습니다. warp당 행 타일 2개가 1개와 4개보다 빨랐습니다(단일 측정): 2048 토큰에서 1115 대 936과 1043 tok/s, 512 토큰에서 1021 대 928과 801. grid x를 평균 타일 수의 1, 2, 4배로 둔 결과는 2% 이내였으므로 상한이 있는 occupancy 조절값으로 2를 택했습니다.

### 3.3 기각: fused NVFP4 decode (측정 후 미반영)

`fused_mamba2_forward`는 affine projection만 곱하므로 NVFP4 export는 unfused Mamba2 step과 unfused MoE 경로를 탑니다(bias가 없으면 `is_affine_quantized`가 false). 이슈는 fused 호출을 확장할 가치가 있는지 물었습니다. PR은 `fused_mamba2_forward`와 `fused_moe_forward` 모두에 NVFP4 지원을 구현하고, 같은 바이너리에서 토글로 3회 측정한 중앙값을 비교했습니다:

| 프롬프트 | Fused | Unfused |
|---|---|---|
| 512 토큰 | 61.0 tok/s | 61.9 tok/s |
| 2048 토큰 | 59.8 tok/s | 60.3 tok/s |

fused 호출은 step당 그래프 구성 시간을 절반으로 줄이지만(약 0.5 대 1.0에서 1.3 ms), 엔진의 decode 루프가 이미 그래프 구성을 이전 step의 GPU 작업과 겹쳐 실행하므로 절약분이 wall clock에 나타나지 않고 fused 쪽이 약간 느립니다. 코드는 버렸고, 수치와 이유는 `src/models/nemotron_h.rs`의 `is_affine_quantized` doc comment와 `docs/installation.md`의 NVFP4 행에 남겼습니다. 다시 검토한다면 먼저 decode가 그래프 구성에 묶이게 되었는지(예: 다른 step당 GPU 시간이 줄어든 뒤) 확인해야 합니다. 절반으로 준 구성 시간이 효과를 내는 조건이 그것입니다.

### 3.4 기본 활성 경로의 수치 변화 수용

스칼라 커널과 WMMA 변형은 둘 다 f32로 누적하지만 순서가 달라 일부 bf16 출력이 한 rounding step만큼 움직입니다. 600 토큰 프롬프트에서 160 토큰 greedy 출력은 약 17 토큰째에서 갈라집니다("repeats about" 대 "repeats the description of"). 짧은 프롬프트는 행 게이트 아래라 바뀌지 않습니다. PR은 이를 결함이 아닌 합산 순서 차이로 받아들였습니다. 테스트는 변형을 스칼라 커널과 같은 per-expert f32 기준과 허용 오차로 검사하고, `MLX_ROCM_NVFP4_GATHER_WMMA=0`이면 이전 출력으로 돌아갑니다. 이 변경 전후의 greedy 출력을 비교하는 사람은 차이를 예상해야 합니다.

## 4. 검증

gfx1151, `Nemotron-Cascade-2-30B-A3B-nvfp4`, `scripts/rocm_gpu_guard.sh --hold` 아래 `scripts/bench_decode.sh --prompt-tokens N`, 3회 중앙값, 토글을 0과 미설정으로 둔 같은 바이너리:

| | 이전 | GPU dequantize만 | 이후 |
|---|---|---|---|
| Prefill 512 | 329.3 | 876.3 | 1007.5 |
| Prefill 2048 | 330.9 | 900.5 | 1105.7 |
| Decode, 512 프롬프트 | 45.2 | 59.9 | 60.0 |
| Decode, 2048 프롬프트 | 43.5 | 59.9 | 58.7 |
| Prefill 64 | 220.6 | | 389.1 |

이득의 대부분은 host gap 제거(3.1)이고, WMMA 변형은 512와 2048 토큰에서 그 위에 약 15에서 23%를 더하며 decode에서는 실행되지 않습니다. `cargo test --features rocm --test rocm_nvfp4_quant`(11 통과, child 1개 ignored)와 `--test rocm_gather_qmm_expert_batched`(2 통과)가 gfx1151에서 통과합니다.

## 5. 학습 포인트

- op 빌더 안의 CPU fallback은 host API 활동이 없는 GPU idle 시간으로 보일 수 있습니다. 트레이스에는 host 스레드에 아무것도 없는 gap이 있었고, 원인은 느린 커널이 아니라 그래프 수준의 stream 우회였습니다. 트레이스에서 어떤 op 주변에 설명되지 않는 idle이 보이면 overlay의 `ops.cpp`에서 `s.device == Device::gpu` 우회를 찾아보십시오.
- APU에서는 fine-grained 메모리 때문에 CPU fallback이 빠른 host 복사를 가진 discrete GPU보다 훨씬 비쌉니다. CPU가 캐시되지 않는 coherent 메모리를 읽기 때문입니다.
- tensor-core 변형은 shape의 합법성만이 아니라 block당 작업량으로 게이트해야 합니다. WMMA 커널은 expert당 3행에서도 합법이지만 18% 느립니다.
- 기각한 최적화는 그것을 다시 시도하게 만들 코드 위치에 수치와 함께 기록합니다. `is_affine_quantized`를 읽는 사람은 NVFP4가 왜 제외되는지, 어떤 조건에서 답이 바뀔 수 있는지 알 수 있습니다.

## 6. 후속 과제와 한계

**미검증: WMMA 커널의 wave64 및 multi-architecture 빌드.** 이 호스트는 gfx1151만 빌드합니다. `qmm.hip`의 `ROCM_HAS_WMMA`는 RDNA3/3.5/4뿐 아니라 CDNA wave64 타깃(gfx908, gfx90a, gfx942, gfx950)에서도 1이므로, fat 빌드에서는 `__half` rocWMMA fragment를 포함한 커널 본문이 wave64 장치용으로도 컴파일됩니다. 기존 `qmm.hip`의 fragment는 `hip_bfloat16`뿐이고, `flash_attention_wmma.hip`은 이미 `__half` fragment를 인스턴스화하므로 컴파일 위험은 줄지만 사라지지는 않습니다. 그 파일은 다른 fragment 구성과 다른 게이트를 쓰기 때문입니다. 런타임에서 wave64 장치는 이 커널을 실행하지 않습니다(`wmma_qmm_wavefront_ok()`가 32 lane wavefront를 요구). 커널의 배치(`warp = tid >> 5`, 32 lane warp 4개가 각자 타일 소유)가 wave32를 전제하고 64 lane wavefront에서는 틀리므로 이 게이트가 중요합니다. 남는 위험은 gfx1151 외 타깃에서의 컴파일 실패(rocWMMA `static_assert` 또는 지원되지 않는 f16 fragment 구성)이며, 그 타깃 목록의 ROCm 빌드 전체를 깨뜨립니다. CDNA 타깃 하나 이상과 gfx1100 또는 gfx1201을 포함한 `MLX_ROCM_ARCHITECTURES` 빌드로 확인할 수 있습니다. `ROCM_HAS_WMMA` 목록 밖의 아키텍처는 빈 본문으로 컴파일되고 dispatch에서 `d.has_native_wmma()`로 제외됩니다.

그 밖의 한계:

- global-scale NVFP4, MXFP4, MXFP8 `dequantize`는 여전히 CPU stream에서 실행됩니다. 임베딩이 이 형식인 체크포인트는 host gap이 남습니다.
- 행 게이트는 expert당 평균 행 수를 쓰며, 크게 치우친 라우팅은 고려하지 않습니다.
- 한 장치에서 한 체크포인트(`Nemotron-Cascade-2-30B-A3B-nvfp4`)만 측정했습니다. 16행 교차점은 다른 RDNA 장치나 다른 expert shape에서 다를 수 있습니다.
- WMMA 경로는 긴 프롬프트의 greedy 출력을 바꿉니다(3.4).

## 참고

- 이슈 #2698, 선행 작업 #2665(expert-batched NVFP4 gather, LOCAL_FIXES item 69), #2617(ROCm 네이티브 NVFP4), item 63(`nvfp4_dequantize_dense`), item 52(`wmma_qmm_wavefront_ok`)
- `src/lib/mlxcelverse/overlays/rocm/LOCAL_FIXES.md` item 70
- `src/models/nemotron_h.rs`(`is_affine_quantized`), `docs/environment-variables.md`, `docs/installation.md`
