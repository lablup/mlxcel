# 기술 보고서: PR #2195 - ROCm 사전 로드 메모리 추정에 in-flight 예산 예약

**날짜**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 헤드 `9374ae72`(origin/main `87538835` 위), PR 열림, 머지 대기. #2155를 닫음(#1801의 일부).

**언어**: Rust(`src/execution/memory_estimate.rs`, 신규 `src/execution/memory_estimate_inflight_tests.rs`, `src/commands/generate.rs`), Markdown(README, `docs/environment-variables.md`, `docs/installation.md`, `docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md`), 텍스트(원시 피크 데이터 파일)

**위험도**: 낮음. 이 변경은 추정치의 예약분만 늘리며, 그것도 ROCm 빌드에서만 그렇습니다. GPU에서 실행되는 내용은 건드리지 않습니다. Metal과 CUDA 빌드는 0을 반환하는 스텁을 컴파일하므로 합계가 바뀌지 않습니다. ROCm에서 보이는 효과는 기본 예산에서 추정치가 1 GiB 커지는 것, 그만큼 paged KV `auto` 예산이 줄어드는 것, 그리고 prompt-cache 용량 상한이 낮아지는 것인데, 이 호스트에서는 선택된 용량이 하나도 바뀌지 않았습니다.

## 요약

PR #2084가 `MLX_ROCM_MAX_INFLIGHT_MB`(기본 1024)로 in-flight 커맨드 배치를 제한한 뒤, ROCm 결과 페이지에는 Meta-Llama-3.1-8B-Instruct-4bit의 640토큰 사전 로드 추정치(5.58)가 측정 피크(6.14 GB)보다 낮다고 기록되었습니다. 추정기에는 커밋됐지만 끝나지 않은 배치가 붙잡을 수 있는 일시 할당, 주로 prefill 중 dequantize-and-GEMM qmm 경로의 f16 가중치 사본을 모델링하는 항이 없었습니다.

이슈 #2155는 그 차이의 대부분이 단위 혼동이며 실제로는 약 0.15 GB만 남는다고 주장했습니다. 이 주장은 틀렸습니다. 유닛이 직접 확인했습니다. 5.58은 `total_bytes / 1e9`이고, `mlxcel inspect`는 같은 추정치를 5.20 GiB로 출력하며, 바이트로는 5,579,686,809입니다. 두 수치 모두 이미 십진 GB였으므로 0.56 GB 차이는 실재했습니다.

수정은 ROCm 빌드에서 in-flight 예산을 독립된 덧셈 항으로 더합니다. `rocm_inflight_reserve_bytes`는 `MLX_ROCM_MAX_INFLIGHT_MB`를 오버레이의 `max_inflight_bytes()`와 정확히 같은 방식으로 파싱하고, 이 항은 `#[cfg(feature = "rocm")]`로 컴파일 시점에 선택됩니다. 이 항은 `total_bytes`와 `runtime_headroom_bytes`에 들어가고, `format_estimate`에서는 `Backend in-flight`로, `inspect --json`과 generate 로그에서는 `backend_inflight_bytes`로 나타나며, `auto_kv_budget_bytes`에서 차감됩니다.

가드를 건 18개 측정(모델 3개, 컨텍스트 2개, 예산 3개)에서 변경 전 추정치는 9개 행에서 피크보다 낮았고, 이제 18개 모두를 덮습니다. 가장 작은 여유는 221,877,772바이트입니다. 테스트 하나가 이 표를 고정하며, 항을 제거하면 실패합니다. `9374ae72`에서 유닛의 `make verify-rocm`은 12,004 통과, 0 실패, 384 무시로 통과했고, 오케스트레이터는 머지 전에 같은 헤드에서 다시 실행했습니다.

## 1. 문제 정의

### 1.1 결과 페이지가 남겨 둔 차이

`estimate_total_memory`는 가중치, 아키텍처 인식 KV, `(DEFAULT_HEADROOM_FACTOR - 1) * (weights + kv)`(계수 1.20, Apple Silicon에서만 보정), 그리고 워크로드 activation 항을 더합니다. ROCm에서는 백엔드가 커밋됐지만 끝나지 않은 배치에 정상 상태 위로 최대 `MLX_ROCM_MAX_INFLIGHT_MB` MiB의 새 할당을 허용합니다(ROCm 오버레이의 `device.cpp`, `LOCAL_FIXES.md` 항목 28). 추정치에는 이에 대한 항이 없었습니다.

`docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md`는 기본 예산에서의 결과를 기록했습니다. 8B 모델은 추정 5.58에 피크 6.14 GB, Qwen3-30B-A3B-4bit은 추정 20.72에 피크 18.58 GB입니다. ROCm용 1.20 계수 재보정은 범위 밖으로 남겨 두었습니다. 예약이 부족한 사전 점검은 로드를 통과시킨 뒤 예산을 넘길 수 있습니다.

### 1.2 이슈에 대한 정정: 단위

이슈의 "Problem / Background" 절은 `bench_decode`가 피크를 `bytes / 1e9`로 출력하고 `format_bytes`는 추정치를 GiB로 출력하므로, 5.58을 GiB로 읽으면 5.99e9바이트이고 추정치는 피크보다 약 0.15 GB만 낮다고 적었습니다.

이 해석은 틀렸고, PR이 그렇게 명시합니다. 결과 페이지의 5.58은 `format_bytes` 값이 아니라 `total_bytes / 1e9`입니다. `mlxcel inspect`는 같은 추정치를 5.20 GiB로 출력하고, 바이트로는 5,579,686,809입니다. 6.14 GB 피크(출력 정밀도 기준 6,140,000,000바이트)와 비교하면 추정치는 약 5억 6천만 바이트 부족했고, 이는 페이지가 보고한 0.56 GB입니다. 그래서 새 결과 절과 PR의 모든 비교는 바이트를 씁니다.

이 정정은 설계에 영향을 줍니다. 0.15 GB 차이라면 계수를 조금 올려 메울 수도 있었습니다. 그러나 기본 예산에서 0.56 GB이고 예산과 함께 커지는 차이(같은 페이지의 스윕에서 8B 피크가 256 MiB와 4096 MiB 사이에서 5.28 GB부터 9.36 GB까지)는 계수가 아니라 in-flight 예산을 가리킵니다.

### 1.3 계수를 키우지 않는 이유

이슈는 ROCm에서 계수를 올리는 방안을 기각했고, PR도 이를 유지합니다. 계수는 가중치에 비례하므로 큰 모델에서 과잉 예약이 됩니다. 30B MoE는 17.17 GB의 가중치 때문에 1.20 계수만으로 3.4 GB가 되어 이미 피크보다 높았습니다. 측정된 초과분은 모델 크기가 아니라 in-flight 예산을 따르므로, 예약은 예산과 같은 덧셈 항입니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/execution/memory_estimate.rs` | 신규 `ROCM_MAX_INFLIGHT_ENV`, `ROCM_DEFAULT_MAX_INFLIGHT_MB = 1024`, `pub fn rocm_inflight_reserve_bytes(raw: Option<&str>) -> u64`와 strtoull을 따르는 보조 함수; `#[cfg(feature = "rocm")]` 본문과 0을 반환하는 `#[cfg(not(feature = "rocm"))]` 스텁을 가진 `backend_inflight_reserve_bytes()`; `MemoryEstimate`와 `InspectReport`에 `backend_inflight_bytes` 필드 추가; 항을 `runtime_headroom_bytes`(따라서 `total_bytes`)에 더함; `auto_kv_budget_bytes`에서 차감; `format_estimate`는 allocator 줄을 계수 항만으로 유지하고 0이 아닐 때 `Backend in-flight` 출력 |
| `src/execution/memory_estimate_inflight_tests.rs`(신규) | 파서 테스트, 오버레이 소스 드리프트 테스트, `auto` 예산 역산 테스트, 두 빌드 종류의 컴파일 시점 게이트 테스트, 포화 테스트, 그리고 18행 gfx1151 표를 고정하는 `measured` 모듈 |
| `src/commands/generate.rs` | `log_estimate_vs_actual_delta`가 `backend_inflight_bytes`를 기록 |
| 문서 | README의 `inspect --json` 필드 목록; `docs/environment-variables.md`(`MLXCEL_HEADROOM_FACTOR` 행, `inspect --json` 문단, `MLX_ROCM_MAX_INFLIGHT_MB` 문단); `docs/installation.md` ROCm 메모리 문단; 결과 페이지의 새 "Estimate against peak" 절 |
| 데이터 | `docs/benchmark_results/data/rocm-memory-gfx1151-2026-09-30/estimate-vs-peak-2026-10-07.txt`, 18회 실행 전부의 가드 `[Memory]` 줄과 피크 줄 원본 |

커밋 세 개: 테스트와 문서를 포함한 추정기 변경(`e387531f`), 어떤 추정치 소비자가 예약분을 보는지에 대한 문서 정정(`160f7e02`), in-flight 테스트의 가드가 UTF-8이 아닌 환경 변수 값을 복원하도록 한 테스트 수정(`9374ae72`). 파일 8개, 추가 661줄, 삭제 13줄.

## 3. 설계

### 3.1 백엔드와 같은 파싱

추정치는 백엔드가 실제로 적용할 값을 예약해야 하므로, `rocm_inflight_reserve_bytes`는 Rust다운 파싱 대신 오버레이 `device.cpp`의 `max_inflight_bytes()`를 그대로 따릅니다.

- 미설정 또는 빈 값: `ROCM_DEFAULT_MAX_INFLIGHT_MB`(1024 MiB).
- 그 외에는 `strtoull(e, &end, 10)` 동작: 앞쪽 C 공백과 선택적 `+` 또는 `-` 하나를 건너뛴 뒤 십진 숫자. 문자열 전체가 소비되고, 첫 바이트가 `-`가 아니며, 숫자가 64비트에 들어가고, MiB 값을 20비트 시프트해도 64비트에 들어갈 때만 값을 채택합니다.
- 그 밖의 값(`1G`, `8abc`, `-1`, 오버플로)은 백엔드처럼 기본값으로 돌아갑니다.
- `0`은 백엔드에서 제한을 끄고, 여기서는 0을 예약합니다. 제한 없는 in-flight 집합은 모델링하지 않습니다. 페이지는 그 상태에서 8B 모델의 피크를 20.60 GB로 측정했고, 환경 변수 문서가 이를 명시합니다.

strtoull의 세부 동작 하나를 의도적으로 재현합니다. 앞쪽 공백 뒤의 `-`는 오버레이의 첫 바이트 검사에 걸리지 않고 strtoull이 부호 없는 산술로 부정하므로, 보조 함수는 `wrapping_neg`를 쓰고 그 결과의 거대한 값은 시프트 검사가 거부하게 둡니다.

`default_budget_and_env_name_match_the_overlay_source`는 `device.cpp`를 읽어 여전히 `constexpr size_t default_max_inflight_mb = 1024;`를 선언하고 `std::getenv("MLX_ROCM_MAX_INFLIGHT_MB")`를 읽는지 확인합니다. 백엔드의 기본값이나 변수 이름이 바뀌면 추정치가 조용히 어긋나는 대신 이 테스트가 실패합니다.

Rust 쪽은 `std::env::var(..).ok()`로 변수를 읽으므로 UTF-8이 아닌 값은 미설정으로 읽혀 기본값을 씁니다. 마지막 커밋은 파서를 바꾸지 않고, 테스트 가드가 테스트 후 그런 값을 정확히 복원하도록 고칩니다.

### 3.2 컴파일 시점 게이트

항은 `src/execution/runtime.rs`의 ROCm cache-limit 기본값과 같은 패턴인 `#[cfg(feature = "rocm")]`로 선택됩니다. 이슈가 런타임 백엔드 질의 대신 이 방식을 요구한 이유는, 추정기가 MLX 상태에 대해 부작용이 없다고 문서화되어 있기 때문입니다. 어느 백엔드인지 알기 위해 런타임을 띄워서는 안 됩니다. ROCm 빌드는 Metal이나 CUDA 백엔드를 컴파일하지 않으므로 빌드 feature가 이미 답을 줍니다. ROCm이 아닌 스텁은 0을 반환하므로 Metal과 CUDA의 합계는 이전과 바이트 단위로 같습니다. `non_rocm_estimate_reserves_nothing_for_inflight`가 그 빌드들에서 이를 확인합니다.

### 3.3 항이 나타나는 곳

- **`total_bytes`와 `runtime_headroom_bytes`.** `runtime_headroom_bytes`는 원래 `weights + kv` 이외의 전체 예약분으로 문서화되어 있었고, 이제 allocator 오버헤드, activation, in-flight의 합이며 `total_bytes`도 이를 따릅니다. `assert_total_is_the_sum_of_its_terms`는 계수 오버헤드를 독립적으로 다시 계산하므로, 합계에서 빠진 항이 headroom 수치 안에 숨을 수 없습니다.
- **`format_estimate`.** allocator 오버헤드 줄은 이제 headroom에서 activation과 in-flight를 모두 빼므로 여전히 계수 항만 보여 줍니다. 별도의 `Backend in-flight: <크기>  (MLX_ROCM_MAX_INFLIGHT_MB)` 줄은 예약분이 0이 아닐 때만 출력되므로, Metal, CUDA, 그리고 `0`으로 설정한 ROCm에서는 새로 출력되는 것이 없습니다.
- **`inspect --json`.** `InspectReport`에 `backend_inflight_bytes`가 생기며, README와 환경 변수 페이지에 ROCm에서만 0이 아니고 `headroom_bytes`의 일부라고 문서화되어 있습니다. 이미 필드를 합산하는 레시피 도구는 `headroom_bytes`와 `total_bytes`가 이를 포함하므로 계속 동작합니다.
- **generate 로그.** `log_estimate_vs_actual_delta`에 `backend_inflight_bytes`가 추가되어 추정 대비 실제 차이를 항별로 나눠 볼 수 있습니다.
- **`auto_kv_budget_bytes`.** 3.4 참조.

`--recommend-quant`는 영향을 받지 않습니다. 추정치의 가중치와 KV 수치만 읽기 때문입니다. 환경 변수 페이지의 첫 개정판은 다르게 적었고, 리뷰 중 커밋 `160f7e02`가 `MLX_ROCM_MAX_INFLIGHT_MB` 문단을 정정했습니다.

### 3.4 auto KV 예산이 예약분을 빼는 이유

`--kv-cache-budget auto`는 추정치의 적합 부등식을 역산합니다. `total = factor × (weights + kv) + activation + inflight`이고 `total ≤ available`이므로 KV에 대해 풀면 `kv ≤ (available − activation − inflight) / factor − weights`입니다. 역산이 새 항을 무시하면, 자신의 추정치가 가용 메모리를 최대 예산만큼 넘는 KV 크기를 고르게 되고, 사전 점검은 auto 정책이 방금 고른 구성을 거부하게 됩니다. `auto_kv_budget_leaves_room_for_the_inflight_reserve`는 고정된 추정치(가용 25e9, activation 1e9, in-flight 1e9, 가중치 10e9이면 9,166,666,666바이트)로 산술을 확인하고, 그 결과의 합계가 들어맞는지도 확인합니다.

### 3.5 포화

파서가 받아들이는 가장 큰 예산은 `u64::MAX >> 20` MiB이고, 바이트 값은 `u64::MAX`에 가깝습니다. headroom과 합계로의 덧셈은 모두 `saturating_add`를 쓰며, `rocm_largest_accepted_budget_saturates_the_total`은 그런 값이 작은 합계로 감싸져 잘못 "들어맞는" 대신 포화되는지 확인합니다.

## 4. 18개 구성 측정

피크는 gfx1151에서 main `97f35bca`에 이 변경을 더한 릴리스 빌드로 다시 측정했습니다(변경은 추정치만 건드리고 실행 내용은 건드리지 않습니다). 각 행은 `scripts/rocm_gpu_guard.sh --idle-secs 30` 안에서 실행한 `mlxcel-bench-decode` 한 번(`-n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens N`, N = 컨텍스트 - 128)이며, 깨끗한 시도에서만 채택했습니다. 4번의 시도가 경합으로 거부되어 재실행되었습니다. 추정치는 `mlxcel inspect --max-tokens <context> --json`에서 가져왔습니다. 피크는 GB 소수 둘째 자리로 출력되므로, 피크 상한은 반올림이 숨길 수 있는 0.005 GB를 더합니다. 모든 수치는 바이트입니다.

| 모델 | 컨텍스트 | 예산(MiB) | 피크 상한 | 변경 전 추정 | 현재 추정 | 현재 - 피크 |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-4bit | 640 | 1024(기본) | 6,145,000,000 | 5,579,686,809 | 6,653,428,633 | 508,428,633 |
| Llama-3.1-8B-Instruct-4bit | 640 | 256 | 5,185,000,000 | 5,579,686,809 | 5,848,122,265 | 663,122,265 |
| Llama-3.1-8B-Instruct-4bit | 640 | 4096 | 9,235,000,000 | 5,579,686,809 | 9,874,654,105 | 639,654,105 |
| Llama-3.1-8B-Instruct-4bit | 4096 | 1024(기본) | 6,955,000,000 | 6,103,135,948 | 7,176,877,772 | 221,877,772 |
| Llama-3.1-8B-Instruct-4bit | 4096 | 256 | 6,025,000,000 | 6,103,135,948 | 6,371,571,404 | 346,571,404 |
| Llama-3.1-8B-Instruct-4bit | 4096 | 4096 | 10,115,000,000 | 6,103,135,948 | 10,398,103,244 | 283,103,244 |
| Qwen3-30B-A3B-4bit | 640 | 1024(기본) | 18,665,000,000 | 20,717,224,703 | 21,790,966,527 | 3,125,966,527 |
| Qwen3-30B-A3B-4bit | 640 | 256 | 17,845,000,000 | 20,717,224,703 | 20,985,660,159 | 3,140,660,159 |
| Qwen3-30B-A3B-4bit | 640 | 4096 | 21,815,000,000 | 20,717,224,703 | 25,012,191,999 | 3,197,191,999 |
| Qwen3-30B-A3B-4bit | 4096 | 1024(기본) | 18,915,000,000 | 21,109,811,558 | 22,183,553,382 | 3,268,553,382 |
| Qwen3-30B-A3B-4bit | 4096 | 256 | 18,785,000,000 | 21,109,811,558 | 21,378,247,014 | 2,593,247,014 |
| Qwen3-30B-A3B-4bit | 4096 | 4096 | 19,265,000,000 | 21,109,811,558 | 25,404,778,854 | 6,139,778,854 |
| Qwen2.5-7B-Instruct-4bit | 640 | 1024(기본) | 5,915,000,000 | 5,240,405,811 | 6,314,147,635 | 399,147,635 |
| Qwen2.5-7B-Instruct-4bit | 640 | 256 | 4,855,000,000 | 5,240,405,811 | 5,508,841,267 | 653,841,267 |
| Qwen2.5-7B-Instruct-4bit | 640 | 4096 | 9,125,000,000 | 5,240,405,811 | 9,535,373,107 | 410,373,107 |
| Qwen2.5-7B-Instruct-4bit | 4096 | 1024(기본) | 6,175,000,000 | 5,469,414,809 | 6,543,156,633 | 368,156,633 |
| Qwen2.5-7B-Instruct-4bit | 4096 | 256 | 5,415,000,000 | 5,469,414,809 | 5,737,850,265 | 322,850,265 |
| Qwen2.5-7B-Instruct-4bit | 4096 | 4096 | 9,455,000,000 | 5,469,414,809 | 9,764,382,105 | 309,382,105 |

**변경 전**: 18개 행 중 9개에서 추정치가 피크 상한보다 낮았습니다. 256 MiB 행을 제외한 모든 dense 행, 그리고 640토큰, 4096 MiB의 MoE 모델입니다.

**변경 후**: 추정치가 18개 모두를 덮습니다. 가장 작은 여유는 221,877,772바이트(Llama-3.1-8B, 4096토큰, 기본 예산)입니다. 각 "현재" 수치는 "변경 전" 수치에 예산의 바이트 값을 정확히 더한 값입니다(예: 5,579,686,809 + 1,073,741,824 = 6,653,428,633). 항이 순수한 덧셈이고 추정치의 나머지는 움직이지 않았음을 보여 줍니다. MoE의 여유가 큰(25억 9천만에서 61억 4천만 바이트) 이유는 계수가 이미 가중치에 대해 과잉 예약하기 때문입니다.

**고정 테스트.** `memory_estimate_inflight_tests::measured::rocm_estimate_covers_every_measured_gfx1151_peak`(ROCm 빌드)는 체크포인트의 config와 가중치 크기로 18개 추정치를 다시 만들고(`inspect` 열과 바이트 단위로 같음), 행의 예산을 설정한 뒤, 추정치 >= 피크 상한을 단언합니다. 또한 항이 없으면 놓쳤을 행 수를 세어 그 수가 0이 아님을 단언하므로, 행 하나가 덮이지 않을 때와 항이 더 이상 의미를 갖지 않을 때 모두 실패합니다. 항을 0으로 강제하면 이 테스트와 다른 테스트 두 개가 실패합니다.

## 5. 부작용: prompt-cache 용량 상한

서버의 모델 인식 prompt-cache 기본값(`src/server/prompt_cache/snapshot_sizing.rs`, snapshot store와 KV store용)은 추정치의 slack(`slack_bytes()`)으로 크기가 정해지며, 상한은 slack의 4분의 1입니다. 이슈는 이 하류 영향을 지적하고 PR에 명시하도록 요구했습니다. ROCm에서는 예약분이 slack을 예산만큼 줄이므로 상한이 예산의 4분의 1만큼 내려갑니다. 기본값에서 268,435,456바이트입니다.

이 호스트(가용 76.80 GiB)에서 대표 길이 8192토큰 기준:

| 모델 | 이전 상한(GB) | 현재 상한(GB) | 대표 엔트리 6개(GB) | 용량 변화 |
|---|---:|---:|---:|---|
| Llama-3.1-8B-Instruct-4bit | 18.93 | 18.66 | 6.44 | 없음 |
| Qwen3-30B-A3B-4bit | 15.22 | 14.95 | 4.83 | 없음 |
| Qwen2.5-7B-Instruct-4bit | 19.18 | 18.91 | 2.82 | 없음 |

용량이 바뀌지 않은 이유는 여기서 상한이 구속하지 않기 때문입니다. 대표 엔트리 6개에 필요한 2.82~6.44 GB는 14.95~18.91 GB 상한보다 훨씬 작으므로, 용량은 상한이 아니라 엔트리 수로 정해집니다. 일반적으로 용량은 상한이 구속하는 경우에만 바뀌며, 그 폭은 예산의 4분의 1과 엔트리 하나 중 큰 쪽 이하이고, 컴파일된 기본값 아래로는 내려가지 않습니다. 메모리가 더 작은 ROCm 호스트나 큰 `MLX_ROCM_MAX_INFLIGHT_MB`에서는 상한이 구속하여 용량이 그 범위만큼 줄 수 있습니다. 이는 의도된 동작입니다. 백엔드가 in-flight 배치용으로 쓸 수 있는 메모리에 캐시 크기를 잡아서는 안 됩니다.

## 6. 검증

gfx1151에서:

- **가드 피크**: 4절의 18개 행, 원본 줄은 `estimate-vs-peak-2026-10-07.txt`.
- **고정 표**: `rocm_estimate_covers_every_measured_gfx1151_peak` 통과, 항을 0으로 강제하면 다른 두 테스트와 함께 실패.
- **대상 테스트**: `cargo test --features rocm --lib -- execution::memory_estimate server::prompt_cache::snapshot_sizing`, 70 통과.
- **린트와 게이트**: lib와 tests에 대한 `-D warnings` clippy, `dead_doc_pointers`, 빠른 `make verify-*` 스크립트 게이트 통과.
- **유닛의 `make verify-rocm`**: `9374ae72`(main `87538835` 위)에서 OK, 153개 테스트 바이너리에서 12,004 통과, 0 실패, 384 무시.
- **오케스트레이터의 `make verify-rocm`**: 머지 전 같은 헤드에서 재실행.

검증하지 못한 것: Metal과 CUDA(이 호스트에 없음). 그 빌드에서의 코드 경로는 0을 반환하는 `#[cfg(not(feature = "rocm"))]` 스텁과, 값이 0인 새 필드 및 JSON 키입니다. `non_rocm_estimate_reserves_nothing_for_inflight`는 여기서 컴파일되거나 실행되지 않았습니다.

## 7. 기술적 선택과 그 이유

- **ROCm용 계수가 아닌 덧셈 항.** 초과분은 가중치가 아니라 in-flight 예산을 따릅니다. 계수를 키우면 이미 과대 추정된 큰 모델에서 과잉 예약이 더 커집니다.
- **백엔드가 적용하는 예산을 정확히 예약.** 잘못된 값의 기본값 복귀까지 포함해 `strtoull`과 오버레이의 채택 검사를 따르면, 추정치와 백엔드가 같은 숫자를 읽습니다. `device.cpp` 대상 드리프트 테스트는 이 일치를 주석이 아니라 검사되는 계약으로 만듭니다.
- **`0`은 아무것도 예약하지 않음.** 제한이 꺼지면 예약할 제한도 없습니다. 추정치는 제한 없는 집합을 모델링하려고 수치를 지어내지 않고, 문서가 그 사실을 밝힙니다.
- **컴파일 시점 선택.** `#[cfg(feature = "rocm")]`는 추정기의 무부작용성을 유지하고, ROCm이 아닌 합계가 구조적으로 바뀌지 않게 합니다.
- **항을 따로 노출.** 별도의 `Backend in-flight` 줄, JSON 필드, 로그 필드로 사용자와 도구가 ROCm 추정치가 왜 큰지 볼 수 있고, allocator 줄은 계수 항만 뜻하게 유지됩니다.
- **auto 예산을 추정치와 일치시킴.** 역산에서 예약분을 빼면 auto 정책이 사전 점검에서 거부될 KV 크기를 고르지 않습니다.
- **공식이 아니라 측정을 고정.** 18행 테스트는 일회성 벤치마크를 회귀 검사로 바꾸고, "의미를 갖는 항" 단언은 테스트가 공허하게 통과하지 않도록 합니다.

## 8. 남은 위험과 후속 작업

- **남은 문서 오류(기존).** `docs/environment-variables.md`의 `MLXCEL_HEADROOM_FACTOR` 행은 여전히 계수를 쓰는 곳으로 `--recommend-quant`를 나열합니다. `--recommend-quant`는 추정치의 가중치와 KV 수치만 읽으므로 계수를 쓰지 않습니다. 이 행은 이 PR보다 오래되었고, PR은 그 행에 문장 하나를 덧붙였지만 소비자 목록은 그대로 두었습니다. 그 행에서 `--recommend-quant`를 빼는 한 줄짜리 후속 작업이 필요합니다.
- **호스트 하나, GPU 세대 하나.** 모든 피크는 gfx1151(RDNA 3.5, 통합 메모리)에서 측정했습니다. 예약분은 설계상 예산과 같으므로 옮겨 갈 것으로 보이지만, 다른 ROCm 타깃에서의 여유는 측정되지 않았습니다.
- **행당 한 번 실행.** 각 피크는 깨끗한 실행 한 번입니다. 가장 작은 여유(4096토큰 8B에서 약 0.22 GB)는 반올림에 비해 넉넉하지만, 피크의 실행 간 편차에 대해서는 입증되지 않았습니다.
- **스윕 밖의 워크로드.** 측정은 640과 4096토큰의 batch 1 디코드를 다룹니다. 더 큰 배치나 컨텍스트는 추정기가 이미 모델링하는 KV와 activation 항을 키우지만, 그 크기에서 in-flight 일시 할당과의 상호작용은 측정하지 않았습니다.
- **`MLX_ROCM_MAX_INFLIGHT_MB=0`.** 이 모드에서 추정치는 피크보다 크게 낮다고 알려져 있습니다(8B에서 20.60 GB 측정). 문서화되었을 뿐 수정되지 않았습니다.
- **Metal과 CUDA는 여기서 미검증.** 스텁은 단순하지만 ROCm이 아닌 테스트는 이 호스트에서 실행되지 않았습니다.

## 9. 학습 포인트

- **단위 주장은 설계하기 전에 바이트로 확인한다.** 이슈의 GiB 해석은 그럴듯해 보였고 문제를 약 4분의 3만큼 줄여 보이게 했을 것입니다. 추정치를 바이트로 출력하자 결론이 났고, 적절한 수정 방향도 바뀌었습니다.
- **예약분이 예산을 따르면 예산을 모델링한다.** 곱셈 계수는 "크기에 비례하는 오버헤드"를 뜻합니다. in-flight 집합은 설정된 숫자로 제한되므로, 그 숫자와 같은 덧셈 항이 더 단순하고 더 정확합니다.
- **예측하려는 구성 요소의 파서를 그대로 따른다.** 백엔드와 다르게 변수를 파싱하는 추정치는 사용자가 가장 설정하기 쉬운 잘못된 값에서 정확히 어긋날 수 있습니다. `strtoull`을 따르고 C++ 소스를 대상으로 테스트하면 둘이 맞춰집니다.
- **공식의 모든 역산은 공식과 함께 바뀌어야 한다.** `auto_kv_budget_bytes`를 고치지 않고 추정치에 항만 더했다면, auto 정책은 자기 사전 점검이 거부하는 구성을 골랐을 것입니다.
- **측정 테스트는 자신이 의미가 있음을 증명해야 한다.** 항이 없으면 어떤 행이 실패함을 단언해야, 보호하려던 것을 제거하는 리팩터링 뒤에도 회귀 테스트가 통과해 버리는 일을 막을 수 있습니다.
