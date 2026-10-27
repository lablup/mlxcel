# PR #2679: ROCm NVFP4 정렬 gather prefill의 expert-batched 경로, Nemotron-H에서 검증

**작성일**: 2026-10-27
**상태**: Linux gfx1151 (ROCm 7.15)에서 완료
**언어**: HIP, Rust, Markdown
**위험도**: 중간 (기본 NVFP4 prefill 경로에 새 커널 추가, Nemotron-H 로더 변경)

## 요약

이슈 #2665는 ROCm의 NVFP4 MoE gather에 expert-batched prefill 경로를 추가하고 실제 체크포인트로 검증할 것을 요구했습니다. 그런데 대상 체크포인트 `mlx-community/Nemotron-Cascade-2-30B-A3B-nvfp4`는 로드조차 되지 않았습니다. PR #2679는 Nemotron-H 로더를 고치고 새 커널을 추가합니다. gfx1151에서 prefill은 512 토큰에서 269에서 328 tok/s로, 2048 토큰에서 237에서 330 tok/s로 올랐습니다. greedy 출력은 커널을 켜든 끄든 바이트 단위로 같습니다.

## 1. 문제

- `GatherQMM::eval_gpu`는 모든 NVFP4 gather를 `nvfp4_qmv_kernel`로 보냈습니다. 이 커널의 multi-row 경로는 8개 행이 weight load를 공유하지만, 16열 block마다 activation 행을 다시 읽습니다. 그래서 prefill은 activation을 약 N / 16번 읽었습니다.
- ROCm에서 실제 NVFP4 체크포인트가 실행된 적이 없었습니다. Nemotron-H의 로컬 `QuantizedSwitchLinear`는 affine `.biases`를 요구하고 `"affine"`을 하드코딩했습니다. 그래서 `Missing switch_mlp.fc1.biases for layer 1` 오류로 로드에 실패했습니다.
- Nemotron-H의 MoE는 prefill을 정렬하지 않았습니다. 정렬된 gather를 위한 커널을 추가해도 이 모델에서는 한 번도 호출되지 않았을 것입니다.

## 2. 변경 내용

| 영역 | 변경 |
|---|---|
| Nemotron-H expert | 로컬 enum을 공용 `SwitchLinear`로 바꿨습니다. 이 타입은 plane에서 mode를 추론하고, 설정 플래그가 없는 `global_scale`은 거부합니다. #958의 선언값 범위 검사는 유지합니다. |
| Nemotron-H MoE | (토큰, expert) 쌍이 64개 이상인 prefill은 mlx-lm의 `SwitchMLP`처럼 expert 순으로 정렬합니다. |
| Fused 경로 | `fused_moe_forward`와 `fused_mamba2_forward`는 affine weight만 곱하므로 affine plane일 때만 쓰도록 제한했습니다. NVFP4는 unfused `gather_qmm`과 SSM 단계를 탑니다. 리뷰에서 지적된 대로, fused MoE는 fc1과 fc2의 packing이 같을 때만 사용합니다. |
| 커널 | `qmm.hip`의 `nvfp4_gather_expert_batched_kernel<T, 8, 4>`는 전치되고 정렬된 `M == 1`, `B >= 64` gather를 처리하며, expert 수 제한이 없습니다. block E는 마지막 expert를 넘어 라우팅된 행을 0으로 채웁니다. `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0`이면 기존 커널을 씁니다. |
| 기록 | LOCAL_FIXES 항목 69, 검증한 체크포인트를 명시한 `docs/installation.md`, `docs/environment-variables.md`. |

## 3. 기술적 결정

### 3.1 run별 dequantize와 GEMM 대신 커널

이슈는 각 expert run을 dequantize한 뒤 GEMM을 호출하는 방식도 제시했습니다. 이 방식은 run 경계를 호스트가 알아야 하므로 호출마다 stream 동기화가 필요합니다. 또 128 x 1856 x 2688 expert에서는 projection 호출마다 bf16 약 1.27 GB를 써야 합니다. GPU만 쓰는 커널은 둘 다 필요 없습니다.

### 3.2 타일 모양

block 하나가 expert 하나를 맡습니다. 각 16-lane 그룹은 activation을 한 번 읽어 여러 열을 계산합니다. 512 토큰에서 측정한 값입니다.

| chunk당 행 | 그룹당 열 | prefill |
|---|---|---|
| 8 | 1 | 290.6 tok/s |
| 8 | 2 | 317.9 tok/s |
| 8 | 4 | 327.1 tok/s |
| 4 | 4 | 312.7 tok/s |
| 16 | 1 | 151.0 tok/s (레지스터 부족) |

채택한 모양은 8행, 4열이며, instantiation은 f32, f16, bf16 세 개만 둡니다.

### 3.3 expert 수 제한 없음

affine 경로는 `E <= 64`와 `B / E >= 4`를 요구합니다. 여기서는 그 제한을 가져오지 않았습니다. 대상 체크포인트는 expert가 128개이고, 512 토큰에서 expert당 약 24행입니다.

## 4. 검증

- **벤치마크**: GPU guard 아래 `scripts/bench_decode.sh`로, 같은 바이너리에서 커널을 끈 경우와 켠 경우를 각각 세 번 돌린 중앙값입니다.
  - 512 토큰: prefill 269.2에서 328.1 tok/s (1.22배).
  - 2048 토큰: prefill 236.5에서 330.3 tok/s (1.40배).
  - decode: 커널을 끄든 켜든 약 44에서 48 tok/s입니다. decode gather는 6행이라 이 경로를 타지 않으므로, 차이는 실행 간 잡음입니다.
  - 짧은 프롬프트: 16 토큰은 114.1에서 119.2 tok/s, 64 토큰은 195.5에서 222.9 tok/s입니다. 128개 expert 대부분이 0에서 2행만 받는 경우인데도 성능이 떨어지지 않았습니다.
- **출력**: 160 토큰 greedy 생성 결과가 커널을 끈 경우와 켠 경우 바이트 단위로 같고, 문장도 자연스럽습니다.
- **커널 테스트**: `rocm_nvfp4_quant` 테스트 8개가 모두 통과합니다. 새로 추가한 128 expert, `B = 512` 정렬 케이스(세 dtype, global scale 유무)도 포함됩니다. `N = 250` 변형은 4열 그룹이 일부만 차는 경우를 다룹니다.
- **Red, global scale**: expert `e + 1`의 global scale을 읽게 하면 global scale 케이스가 실패합니다(상대 오차 0.53에서 0.54).
- **Red, expert offset**: expert `e + 1`의 weight를 읽게 하면 모든 정렬 케이스가 실패합니다(상대 오차 1.40에서 1.58).
- **로더 테스트**: 새 Nemotron-H lib 테스트는 NVFP4 plane을 로드하고 정렬, 비정렬 MoE 경로를 모두 확인합니다. `main`의 로더에서는 biases 누락 오류로 실패합니다. nemotron_h lib 테스트 74개가 모두 통과합니다.
- **Lint**: `-D warnings` clippy, fmt, overlay records가 통과합니다.

## 5. 남은 한계

- **남은 prefill 시간**: 512 토큰 prefill의 kernel trace에서 새 커널은 호출당 7.7 ms로, 전체 시간의 약 23%입니다. 전체 시간의 약 절반 동안 GPU가 놀고 있는데, 이는 이 PR이 다루지 않은 호스트 측 비용입니다. WMMA 타일 커널로 MoE 비중을 더 줄일 여지가 있습니다.
- **NVFP4 decode**: fused 단계가 affine 전용이므로, Nemotron-H는 unfused Mamba2 단계로 decode합니다.
- **체크포인트 범위**: 측정한 NVFP4 체크포인트는 하나뿐입니다. 대체 후보 `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`는 내려받지 않았습니다.
- **다른 백엔드**: Nemotron-H prefill 정렬은 Metal과 CUDA에도 적용되지만, mlx-lm과 같이 non-fused 경로(latent projection 모델이나 fused port가 없는 백엔드)에서만 동작합니다. 해당 백엔드에서는 측정하지 않았습니다.
