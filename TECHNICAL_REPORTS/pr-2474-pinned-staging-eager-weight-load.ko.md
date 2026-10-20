# PR #2474: Pinned 메모리 스테이징과 eager 가중치 로드

**작성일**: 2026-10-20
**상태**: 완료 (PR 열림 상태, 미검증 항목은 5절 참고)

## 요약

PR #2474(#1564 종결)는 가중치 업로드를 첫 토큰에서 제거한다. `Load::eval_gpu`가 재사용하는 64 MiB pinned host slot 4개를 거쳐 텐서를 올리고, 모든 loader의 텍스트 가중치는 `Model loaded` 출력 전에 materialize된다. GB10에서 `qwen3.8-27B-4bit` 기준 첫 토큰은 4.02 s에서 0.67 s로, 로드 + 첫 토큰은 4.37 s에서 2.24 s로 줄었고, decode 속도는 그대로이며 peak 메모리 증가는 0.1 GB 미만이다.

## 1. 문제 정의

MLX `Load` 배열은 lazy다. `load_safetensors`는 평가되지 않은 배열을 반환하므로 체크포인트는 해당 배열이 처음 필요한 `eval` 안, 즉 첫 토큰에서 읽히고 디바이스로 복사된다. `mlxcel generate`는 resident 0.00 GB 상태로 `Model loaded`를 출력한 뒤 업로드 전체를 첫 토큰에 부과했다(GB10에서 첫 토큰 4.02 s 중 3.35 s, #1545 기록의 V100에서는 78%). 원인은 두 가지로 분리된다.

- **`Load::eval_gpu`의 스테이징.** 텐서마다 pageable `malloc`, `pread`, 동기식 pageable `cudaMemcpyAsync`, `free`를 거쳤다. GB10에서 DMA 자체는 어느 경로든 0.26 s(약 58 GB/s)로 같았고, 비용은 호스트 시간이었다. `cudaMemcpyAsync` API 시간 0.89 s, CUDA 프로파일러에 보이지 않는 호스트 스테이징 구간 2.20 s(4.7-5.3 GB/s)이다.
- **정책.** loader가 materialization을 로드에 부과할지 첫 토큰에 부과할지는 loader별 우연이었다(Gemma 4 loader만 가중치를 평가). 서버는 첫 요청에서 업로드 비용을 치렀다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/load.cpp`, `load_staging.h` | 디바이스당 64 MiB pinned slot 4개를 쓰는 `Load::eval_gpu` overlay |
| `src/lib/mlxcelverse/overlays/rocm/mlx/backend/rocm/load.cpp`, `load_staging.h`, `mlx/io/safetensors.cpp` | ROCm `Load::eval_gpu` 동등 구현 (디바이스당 64 MiB `hipHostMallocPortable` slot 4개, 업로드 카운터, `load_safetensors` GPU stream 배정) |
| `src/lib/mlxcel-core/src/weight_materialize.rs` | `MaterializationScope`, 정책 파싱, tower 텐서 제외 |
| `src/lib/mlxcel-core/cpp/mlx_cxx_ext.cpp`, `mlx_cxx_bridge.h`, `build.rs` | `array_weak_ref` / `array_weak_upgrade` 및 CUDA/ROCm `weight_load_transfer_stats` bridge 함수 |
| `src/loading/materialize.rs`, `src/loading/mod.rs` | `load_model`과 adapter 변형이 scope를 열고 종료 |
| `src/lib/mlxcel-core/src/weights.rs` | weights loader가 반환 텐서를 열린 scope에 등록 |
| `generate.rs`, `engine/direct.rs`, `ttft_profile_tests.rs` | `[TTFT] load` 줄, 첫 토큰 줄에 업로드 카운터 추가 |
| 테스트 | `weight_materialize_tests.rs`(355줄), `ttft_profile_tests.rs` |
| 문서 | 벤치마크 기록, `docs/environment-variables.md`, `CHANGELOG.md`, overlay README |

18개 파일, +1582/-7. 커밋: `e56456258`(feat), `054792a5b`(벤치마크 기록), `4a23cfbfd`(weak reference), `8b5179819`(chunking 테스트와 오류 호출명 표기).

## 3. 기술적 선택과 그 이유

### 3.1 `Load::eval_gpu`의 pinned slot ring

**동작.** 디바이스당 64 MiB `cudaHostAlloc` slot 4개를 한 번 할당해 재사용한다. 한 slot으로의 read가 이전 slot에서 나가는 `cudaMemcpyAsync`와 겹치고, slot마다 CUDA event가 있어 복사가 끝난 뒤에만 재사용된다. chunk 크기는 `min(64 MiB, nbytes - done)`이므로 slot보다 큰 텐서(embedding과 untied `lm_head`, 각 약 636 MB)는 복사 10회가 된다. chunk 경계가 64 MiB의 배수라 endianness swap이 원소를 가르지 않는다. 텐서마다의 `malloc`, 첫 접근 page fault, 체크포인트 매핑을 해제하던 `free`가 사라졌다.

**실패 처리.** `cudaHostAlloc` 또는 event 생성이 실패하면 실패한 호출 이름과 함께 한 번 경고하고 upstream pageable 경로로 돌아간다. `MLXCEL_CUDA_PINNED_LOAD=0`은 이 경로를 명시적으로 선택한다. slot은 해제하지 않으므로 진행 중인 복사가 해제된 메모리를 읽지 않는다.

**트레이드오프.** 디바이스당 256 MiB의 pinned host 메모리를 MLX allocator 밖에서 쓴다. 호출 수는 1,847에서 1,865로 늘어난다(chunk 18개 추가).

### 3.2 모든 loader에 하나의 materialization 정책

**대안.**

| 옵션 | 장점 | 단점 |
|---|---|---|
| loader마다 자체 평가 | 공유 상태 없음 | 이미 불일치(Gemma 4만 해당), 새 family마다 기억해야 함 |
| `weights.rs` loader 안에서 평가 | 한 곳 | loader는 모델이 무엇을 유지하고 버리는지 모름 |
| **`load_model`이 scope를 열고 loader가 텐서를 등록, `finish`가 살아남은 텐서를 평가** | 모든 family가 정책을 상속, 완성된 모델 기준 | registry와 수명 규칙 필요 |

`finish`는 등록된 텐서 중 modality tower와 버려진 텐서를 제외하고 `eval_all` 후 `detach_all`을 호출한다. `MLXCEL_WEIGHT_MATERIALIZATION=lazy`는 첫 사용 시 materialization으로 되돌린다. scope 밖의 로드(pipeline stage, embedding loader, 테스트)는 등록하지 않으므로 lazy로 남는다.

### 3.3 strong handle 대신 weak reference (리뷰 중 변경)

첫 버전은 등록된 모든 텐서에 strong handle을 두고 `array_use_count`로 버려진 텐서를 판별했다. 리뷰(HIGH)에서, 로드 중 repack을 평가하고 원본을 버려 peak를 제한하는 loader(`fp8_block.rs`의 FP8 block requantization, `kimi_k3_sanitize.rs`의 expert stacking)에서 모든 원본 버퍼가 로드 끝까지 남아 peak가 대략 2배가 된다는 점이 발견됐다.

registry는 이제 `array_weak_ref` / `array_weak_upgrade`로 weak reference를 둔다. 등록이 텐서 수명을 늘리지 않고, 버려진 원본은 loader가 버리는 즉시 해제되며, 아무도 잡고 있지 않은 텐서는 null로 upgrade되어 건너뛰므로 읽히지 않는다. moved-from handle도 null로 upgrade되어 미참조로 센다. 보안 리뷰는 bridge를 확인했다. `array`는 `array_desc_` 멤버 하나뿐이고 교체되는 descriptor는 정적 placeholder의 것(use count > 1)이라 `array::release`를 우회해도 안전하다. 버려진 텐서가 해제되는지 검증하는 회귀 테스트가 있다.

### 3.4 vision/audio tower는 lazy 유지

`is_multimodal_tower_tensor`는 `vision_tower.`, `visual.`, `audio_tower.` 등 선행 이름과 tower에만 입력을 넘기는 projector를 판별한다. 제외하지 않으면 텍스트 전용 실행이 쓰지 않는 메모리를 점유한다. 벤치마크 체크포인트에서 2,180개 중 333개가 lazy로 남고, `language_model.` 텐서 1,847개(15.13 GB)가 로드 후 resident다. tower 텐서는 첫 이미지나 클립에서 로드된다.

### 3.5 lazy로 남는 것

- tensor-parallel(`--tp-size` > 1), pipeline-stage, embedding/reranker 로드: 텐서를 등록하지 않는다.
- loader가 버린 텐서(MTP head, tied `lm_head`, sanitize로 제거된 중복).
- `Load` 배열에서 파생된 텐서(복사, repack, dtype 변환): 원본은 materialize되고 파생 연산은 첫 토큰에서 실행되며, MLX가 결과가 만들어질 때마다 원본을 해제한다.

## 4. 측정 결과 (GB10)

`qwen3.8-27B-4bit`, 2토큰 프롬프트, page cache와 JIT cache 모두 warm, 6회 중앙값(교차 2라운드). 전체 기록은 `docs/benchmark_results/weight-materialization-gb10-2026-10-20.md`.

| 지표 | 변경 전 | 변경 후 |
|---|---|---|
| 모델 로드 | 0.35 s | 1.58 s (15.13 GB resident) |
| 첫 토큰 | 4.02 s | 0.67 s |
| 로드 + 첫 토큰 | 4.37 s | 2.24 s (-49%) |
| 프로세스 wall | 5.44 s | 3.30 s |
| `cudaMemcpyAsync` 호스트 API 시간 (nsys) | 0.89 s / 1,847회 | 0.02 s / 1,865회 |
| 호스트 스테이징 구간 | 2.20 s | 0.90 s |
| Decode 기울기 (-n 40 대 120) | 75.6 ms/token | 76.1 ms/token (반복 편차 1.4 ms 이내) |
| 첫 prefill 후 MLX peak | 15.54 GB | 15.57-15.60 GB |

arm별 기여: lazy + pinned만 쓰면 로드 + 첫 토큰이 2.229 s, eager + pageable은 2.635 s, eager + pinned는 2.242 s다. pinned 스테이징이 4.37 s 중 약 2.15 s를 제거한다. eager는 남은 1.25 s의 업로드를 첫 토큰에서 로드로 옮길 뿐 줄이지 않는다. 두 스위치를 모두 끈 arm은 기준선을 0.4% 이내로 재현한다.

같은 프로세스의 두 번째 prefill은 0.14 s이므로 첫 토큰 앞의 일회성 비용은 변경 전 3.88 s, 변경 후 0.53 s다. 남은 0.53 s는 가중치 I/O가 아니라(`eval weights=0`) JIT 모듈 로드, CUDA graph 인스턴스화, stream 간 fence다.

OLMoE-1B-7B(raw bf16, expert 텐서 3,072개 전부를 loader가 재적층): peak 13.93 GB에서 14.14-14.40 GB(+1.5-3.4%). MLX가 repack 결과가 만들어질 때 원본을 해제하므로 단순 추정의 2배는 아니다.

정확성: greedy 출력(`-n 24 --temp 0`)은 변경 전, eager + pinned, eager + pageable, lazy 사이에서 바이트 단위로 동일하고, 이미지 요청도 두 바이너리에서 같다.

### 4.1 ROCm (`gfx1151`, AMD Radeon 8060S Graphics) 측정 결과

`gfx1151`에서 PR을 검증하는 과정에서 ROCm 전용 격차 2건을 확인하고 함께 해결했다.
1. **ROCm overlay의 `mlx/io/safetensors.cpp`가 기본 `Load` stream 선택 시 `cu::is_available()`만 검사.** ROCm에서는 `load_safetensors`가 `Device::cpu`(`Load::eval_cpu`)로 빠져 `Load::eval_gpu`를 우회하고, loader가 `Load` 배열로부터 파생 텐서를 만들 때 CPU→GPU stream 간 fence가 발생했다. `cu::is_available() || rocm::is_available()`로 수정해 `Load`를 `Device::gpu`에 스케줄한다.
2. **`mlx/backend/rocm/load.cpp`의 텐서별 `hipHostMalloc` + `hipLaunchHostFunc(hipHostFree)`.** eager `eval_all` 아래에서는 텐서마다 페이지 pinning이 반복되면서 `gfx1151` 기준 2 GiB당 413 ms의 KFD 페이지 pinning 비용이 발생했고(16 × 128 MiB = 2 GiB 단독 HIP 벤치마크: 총 617.22 ms — `hipHostMalloc` 297.26 ms, `pread` 156.37 ms, `hipMemcpyAsync` 47.67 ms, `hipHostFree` 115.91 ms — 대 4 × 64 MiB `StagingPool` 재사용 시 **162.98 ms**, **3.79배 단축**), 배치 완료 전까지 디바이스 가중치와 중복되는 pinned 스테이징 버퍼가 메모리에 남았다. `mlx/backend/rocm/load.cpp`와 `load_staging.h`도 CUDA와 동일하게 4 × 64 MiB pinned `StagingPool`(`hipHostMallocPortable` + `hipEventDisableTiming`)과 `LoadTransferStats` 카운터를 사용하며, `MLXCEL_ROCM_PINNED_LOAD=0`(또는 `MLXCEL_CUDA_PINNED_LOAD=0`)으로 pageable fallback을 선택할 수 있다.

`gfx1151` 종단간 warm-cache 측정 결과 (`--no-chat-template -p "The capital of France is" -n 16 --temp 0 --seed 0 --profile`, 프롬프트 5토큰):

| 모델 | Arm | 모델 로드 | 첫 prefill (TTFT) | 로드 + 첫 토큰 | Decode | 업로드 카운터 (`[TTFT]`) |
|---|---|---|---|---|---|---|
| `Qwen3-0.6B-4bit` (텐서 704개, 0.34 GB) | `eager + pinned` | 0.304 s (resident 0.31 GB) | **397.61 ms** | 0.702 s | 303.35 tok/s | load `eval=85.66ms pinned=704 memcpy=705`, prefill `weights=0` |
| `Qwen3-0.6B-4bit` | `lazy + pinned` | 0.169 s (resident 0.00 GB) | 470.86 ms | 0.640 s | 318.86 tok/s | prefill `weights=704/0.34GB upload=68.34ms pinned=704` |
| `Qwen3-VL-2B-Instruct-4bit` (텍스트 704개 / 비전 제외 315개, 0.97 GB) | `eager + pinned` | 0.285 s (resident 0.90 GB) | **395.79 ms** (-23%) | 0.681 s | 145.15 tok/s | load `eval=108.40ms pinned=704 memcpy=706`, prefill `weights=0` |
| `Qwen3-VL-2B-Instruct-4bit` | `lazy + pinned` | 0.169 s (resident 0.00 GB) | 515.05 ms | 0.684 s | 143.63 tok/s | prefill `weights=704/0.97GB upload=105.01ms pinned=704` |
| `Meta-Llama-3.1-8B-Instruct-4bit` (텐서 743개, 4.52 GB) | `eager + pinned` | 0.555 s (resident 4.21 GB) | **444.79 ms** (lazy+pinned 대비 -46%, lazy+pageable 대비 -50%) | 1.000 s | 43.42 tok/s | load `eval=407.59ms staging=313.19ms pinned=743 memcpy=749`, prefill `weights=0` |
| `Meta-Llama-3.1-8B-Instruct-4bit` | `eager + pageable` | 0.610 s (resident 4.21 GB) | 449.13 ms | 1.059 s | 43.27 tok/s | load `eval=459.69ms pageable=743 memcpy=743` |
| `Meta-Llama-3.1-8B-Instruct-4bit` | `lazy + pinned` | 0.154 s (resident 0.00 GB) | 826.21 ms | 0.980 s | 43.19 tok/s | prefill `weights=743/4.52GB upload=373.87ms pinned=743 memcpy=749` |
| `Meta-Llama-3.1-8B-Instruct-4bit` | `lazy + pageable` | 0.151 s (resident 0.00 GB) | 883.64 ms | 1.035 s | 43.61 tok/s | prefill `weights=743/4.52GB upload=479.22ms pageable=743` |
| `Qwen3-30B-A3B-4bit` (텐서 1,351개, 17.17 GB) | `eager + pinned` | **9.409 s** (resident 16.00 GB) | **634.01 ms** (lazy+pinned 대비 -94.0%, lazy+pageable 대비 -95.3%) | **10.043 s** (lazy+pageable 대비 -26.9%) | 61.37 tok/s | load `eval=9175.45ms staging=8537.43ms pinned=1351 memcpy=1499`, prefill `weights=0` |
| `Qwen3-30B-A3B-4bit` | `eager + pageable` | 13.610 s (resident 16.00 GB) | 988.03 ms | 14.598 s | 61.06 tok/s | load `eval=13424.21ms pageable=1351 memcpy=1351` |
| `Qwen3-30B-A3B-4bit` | `lazy + pinned` | 0.188 s (resident 0.00 GB) | 10,552.30 ms | 10.740 s | 59.27 tok/s | prefill `weights=1351/17.17GB upload=9908.50ms pinned=1351 memcpy=1499` |
| `Qwen3-30B-A3B-4bit` | `lazy + pageable` | 0.185 s (resident 0.00 GB) | 13,555.83 ms | 13.741 s | 33.16 tok/s | prefill `weights=1351/17.17GB upload=12985.53ms pageable=1351` |

`gfx1151`에서 테스트한 모든 체크포인트(`Qwen3-0.6B-4bit`, `Qwen3-VL-2B-Instruct-4bit`, `Meta-Llama-3.1-8B-Instruct-4bit`, `gpt-oss-20b-MXFP4-Q4`, `Qwen3-30B-A3B-4bit`)에서 `eager`와 `lazy`의 greedy 출력(`-n 16 --temp 0 --seed 0`)은 바이트 단위로 동일하다.

## 5. 미검증 및 미해결

- **V100/PCIe 디바이스 없음.** 이슈의 V100 수치는 재측정하지 않았다. GB10과 `gfx1151`은 coherent/unified memory이므로 개별 PCIe 디바이스의 이득은 다를 수 있다.
- **Metal.** 기본값이 eager인 것은 같고(mlx-lm 기본 로드 동작과 일치) 실행하지 않았다. pinned `Load::eval_gpu` overlay는 CUDA와 ROCm을 모두 지원한다.
- **Multi-GPU, tensor-parallel.** 단일 GPU 호스트라 실행하지 못했다. 프로세스 전역 mutex 하나가 모든 디바이스 pool을 감싸므로 여러 GPU를 구동하는 프로세스는 업로드가 직렬화된다(LOW).
- **실패 주입.** pinned 할당 실패는 주입하지 않았고 fallback은 `MLXCEL_CUDA_PINNED_LOAD=0` / `MLXCEL_ROCM_PINNED_LOAD=0`으로만 실행했다. `ensure_ready`는 `cudaHostAlloc` / `hipHostMalloc` 실패 시 `slot.host`가 null로 남는다는 점에 의존한다(LOW). 복사 성공 후 `cudaEventRecord` / `hipEventRecord`가 실패하면 slot이 pending으로 표시되지 않는데, 이미 eval을 실패시키는 GPU 오류 이후에만 도달한다(LOW).
- **남은 첫 토큰 비용.** `event_signal_kernel`이 eager 포함 모든 arm의 첫 토큰에서 1,847회 실행된다. `Load` 배열에서 파생된 가중치가 첫 토큰에서 generation stream으로 실행되기 때문이며, 남은 0.53 s 안의 다음 후보로 이번 PR에서는 다루지 않았다.
- **기타.** slot event는 event 자체의 blocking-sync 플래그 없이 `cudaEventDisableTiming` / `hipEventDisableTiming`만 써서 CUDA에서는 slot 대기가 코어 하나를 spin한다. `MLXCEL_CUDA_PINNED_LOAD` / `MLXCEL_ROCM_PINNED_LOAD`는 대소문자를 구분하고 `no`를 받지 않는다. 이 PR 이전부터 upstream에 있는 문제: pageable 경로의 `malloc`은 검사하지 않고 read 예외 시 누수되며, `safetensors.cpp`는 `offset + data_offsets[1]`을 오버플로 검사 없이 더한다.
- 같은 코드인데 GB10에서 pageable read가 첫 `eval` 안(2.20 s)보다 로드 시점(eager, 0.95 s)에 더 싼 이유는 조사하지 않았다. 기본값에는 영향이 없다.

## 6. 검증

- CUDA(GB10)와 ROCm(`gfx1151`) 양쪽에서 `weight_materialize_tests`(10/10)와 `ttft_profile_tests`(7/7): 버려진 텐서가 해제되는지 검증하는 테스트와 `pinned_upload_chunks_and_wraps_the_slot_ring`(352 MiB 텐서, chunk 6개, slot 4개 ring)를 포함한다. 후자는 chunking 경로가 바이트 동일 출력의 636 MB embedding으로만 검증된다는 리뷰 지적 후 `8b5179819`에서 추가됐다.
- `cargo test --release --features cuda -p mlxcel --lib -- loading::`(372개 통과) 및 `cargo test --profile test-fast --features rocm -p mlxcel --lib -- loading::`(371개 통과, ROCm 미지원 native `nvfp4` 1건 제외).
- `scripts/ci/rocm_smoke.sh`(통과, 234.92 tok/s) 및 `make verify-rocm-overlay`(records + drift + ownership mock 통과).
- mlxcel-core와 mlxcel lib/tests(CUDA 및 ROCm)에 `cargo clippy -D warnings`, `cargo fmt`, 라이선스 헤더 검사.
- 리뷰: 구현 리뷰(HIGH 1건, `4a23cfbfd`에서 수정), 보안/성능 리뷰는 CRITICAL/HIGH 없음. 확인된 항목: slot 경계, 신뢰할 수 없는 체크포인트 offset(overlay는 upstream과 같은 바이트 범위를 읽음), 예외 안전성(`ParallelFileReader::read`가 던지기 전에 모든 batch를 join), `cudaEventSynchronize` / `hipEventSynchronize`로 인한 deadlock 없음.

## 7. 관련 항목

- 이슈 #1564(이 PR이 종결). `docs/benchmark_results/volta-ttft-fixed-cost-2026-09-01.md`(#1545)가 V100 기준선을 측정하고 세 후보를 제시했다.
- 새 환경 변수: `MLXCEL_WEIGHT_MATERIALIZATION`(기본 `eager`, `lazy`), `MLXCEL_CUDA_PINNED_LOAD`(`0`이면 비활성), `MLXCEL_ROCM_PINNED_LOAD`(`0`이면 비활성). `MLXCEL_PROFILE_TTFT`에 `[TTFT] load` 줄과 업로드 카운터가 CUDA/ROCm 모두에 추가됐다.
