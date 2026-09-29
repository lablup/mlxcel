# 기술 보고서: PR #2056 - 벤치마크 하니스의 ROCm 지원과 gfx1151 기준선

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: Bash (벤치마크 하니스), Python (mlx-lm 하니스, CSV 비교, 테스트), Markdown, CSV

**위험도**: 낮음 (벤치마크 스크립트, 문서, CSV, Python 테스트만 바뀌었고 Rust, C++, 커널, 빌드 경로는 바뀌지 않았습니다. `bench_decode.sh`의 Metal과 CUDA 경로는 순서만 바뀌었고 출력은 같지만, 해당 호스트에서 실행해 보지는 않았습니다)

## 요약

이 PR 이전에는 AMD Strix Halo 호스트에서 `scripts/bench_decode.sh`를 돌리면 ROCm 결과가 `benchmarks/metal_<CPU 이름 앞 20자>_<date>.csv`로 저장되었을 것이고, 메모리 부족 가드는 GPU가 쓰는 96 GiB carve-out이 아니라 호스트가 보는 31 GiB를 기준으로 잡혔을 것입니다. CSV에는 upstream MLX pin만 기록되어 upstream MLX와 ROCm overlay를 포함한 빌드를 구분할 수 없었습니다. `scripts/compare_bench_csv.py`는 `m5max`와 `m1ultra` 호스트만 알았기 때문에 그 밖의 모든 호스트에서 superseded-baseline 경고가 꺼져 있었습니다. 이슈 #1810(에픽 #1801의 Phase 3)은 올바르게 표시되고 비교 가능한 ROCm 결과와 첫 공개 기준선을 요구했습니다.

이 PR은 두 하니스가 ROCm 호스트를 인식하게 하고, ROCm 실행을 `benchmarks/rocm_strixhalo-gfx1151_<date>.csv`로 저장하며, 메모리 예산을 device memory 기준으로 잡고, ROCm 전용 출처 컬럼 두 개(`mlx_rocm_overlay_commit`, `hip_version`)를 추가하고, 비교 스크립트가 어떤 하니스 파일명에서도 runtime과 host를 읽도록 합니다. 또한 첫 gfx1151 처리량 기준선을 공개합니다. 체크포인트 다섯 개를 pp512/tg128로, 같은 호스트와 같은 날에 mlxcel과 mlx-lm을 비교했습니다. 커밋된 CSV 기준으로 mlxcel decode 속도를 mlx-lm으로 나눈 값은 Qwen3-0.6B-4bit 1.27x(278.48 대 220.13 tok/s), Llama-3.1-8B-Instruct-4bit 1.08x(35.41 대 32.81), Qwen3-30B-A3B-4bit 1.05x(61.82 대 58.65), gpt-oss-20b-MXFP4-Q4 1.00x(8.07 대 8.07)입니다. Mixtral-8x7B의 decode는 두 런타임 모두에서 실행마다 약 8에서 11 tok/s 사이로 달라지므로 비율을 주장하지 않습니다.

## 1. 문제 정의

### 잘못된 표시와 잘못된 메모리 예산

이슈 #1810은 `d8d34e2b`에서 각 문제를 다시 확인했습니다.

- `detect_backend`는 `nvidia-smi`가 동작하거나 `generate --help`에 cuda가 나올 때만 `cuda`를, 아니면 `metal`을 반환했습니다. ROCm sweep은 `metal_` 아래에 저장되었을 것입니다.
- `nvidia-smi`가 GPU를 찾지 못하면 하드웨어 태그는 `/proc/cpuinfo`의 model name 앞 20자로 대체되었습니다. 커널을 실행한 장치가 아니라 CPU 이름입니다.
- 메모리 가드는 `free -b`의 85%를 사용했습니다. UMA carve-out에서는 약 세 배 차이가 나는 잘못된 값입니다. 호스트는 약 31 GiB를, GPU는 96 GiB를 봅니다. GPU에 여유 있게 들어가는 26 GB Mixtral이 `SKIP:oom_estimate`로 건너뛰어졌을 것입니다.
- `mlx_commit`은 8자리 upstream pin입니다. ROCm 빌드는 NripeshN/mlx `rocm-support`를 그 pin 위로 옮기고 로컬 수정을 더한 것인데, 행 어디에도 그 사실이 없었습니다.
- `compare_bench_csv.py`는 `m5max`와 `m1ultra`를 하드코딩하고 `pylm`이 아닌 모든 runtime을 `metal`로 취급했습니다. 다른 호스트(GB10, V100, gfx1151)에서는 `newer_readings_elsewhere`가 `{}`를 반환했으므로, 스크립트 docstring이 가장 중요하다고 적은 검사가 조용히 꺼져 있었습니다.

### 모델을 로드하지 않고는 장치를 알 수 없음

이슈의 계획은 #1805(PR #1883)의 `gpu_backend_kind()`를 권위 있는 backend 출처로 지목했지만, 하니스는 셸 스크립트이고 그 값을 출력하는 CLI 서브커맨드는 없습니다. `generate --help`는 백엔드와 관계없이 같습니다. backend, `gfx` target, device memory를 알려 주는 CLI 출력은 실제 `mlxcel generate` 실행뿐입니다. `MLXCEL_DEBUG_KERNEL_BACKEND=1`일 때 stderr의 `[mlxcel] custom kernel backend: rocm`, stdout의 `HIP architecture gfx1151; compiled for [...]`와 `GPU: <name> (Amd), <N> GiB device memory.`입니다. `scripts/ci/rocm_smoke.sh`가 이미 이 줄들을 파싱하고 있었습니다.

### 기준선 없음

`benchmarks/`와 `docs/benchmark_results/` 어디에도 ROCm 처리량 데이터가 없었습니다. 이슈에는 참고용 spike 수치만 있었고, #1814의 ROCm 성능 작업에는 프로젝트 자체 하니스로 측정한 재현 가능한 기준이 필요했습니다.

## 2. 변경 요약

- **`scripts/bench_decode.sh`** (283줄 변경).
  - *런타임 probe.* `probe_runtime`은 Linux에서 `nvidia-smi`가 실패할 때만 실행됩니다. `MODELS_DIR`에서 safetensors 크기가 가장 작은 체크포인트(`smallest_checkpoint`)를 고르고, 저장소가 비어 있을 때만 지정된 모델을 쓰며, `BENCH_PROBE_TIMEOUT`(기본 300초) 안에서 `MLXCEL_DEBUG_KERNEL_BACKEND=1 mlxcel generate -m <model> -p Hello -n 1`을 실행합니다. `sed` 파서 네 개가 출력에서 backend, `gfx` target, 장치 이름, device memory를 읽습니다.
  - *대체 경로.* backend 줄을 전혀 출력하지 못한 probe(커널이 결정되기 전에 실패한 경우)는 ROCm이 아니라는 증거로 쓰지 않고, `rocminfo`가 첫 GPU agent의 `gfx` 이름과 marketing name을 제공합니다. `rocminfo`는 KFD 드라이버가 멈추면 블록될 수 있어 60초 timeout 안에서 실행됩니다. 다른 backend를 출력한 probe는 그대로 믿고 `rocminfo`를 보지 않습니다. device memory는 `/sys/class/drm/card*/device/mem_info_vram_total`로 대체되고, ROCm 릴리스는 `/opt/rocm/.info/version` 또는 `core*/.info/version`에서, HIP 버전은 `hipconfig --version`에서 읽습니다. 이 helper들은 `set -euo pipefail` 아래의 command substitution에서 돌기 때문에, 실패하지 않고 빈 필드를 반환합니다.
  - *표시.* `detect_backend`는 NVIDIA 검사 뒤, `--help` 검사 앞에서 `rocm`을 반환합니다. `detect_hardware_full`은 CUDA 문자열과 같은 모양으로 `<장치 이름>_<gfx>_ROCm<릴리스>_<device memory>GB`를 만듭니다. `detect_hardware_short`는 `*_gfx1151_*`을 `strixhalo-gfx1151`로, 다른 `gfx` target은 `amd-<gfx>`로 바꿉니다. 20자 절단을 쓰면 target이 잘려 나가므로 쓰지 않습니다.
  - *예산.* `detect_memory_bytes`는 backend가 `rocm`이고 장치 수치를 찾았으면 device memory를, 아니면 host memory를 반환합니다. 스크립트는 `ROCm: <릴리스> (HIP <v>), <gfx>`와 `Memory budget: 85% of <N> GiB (<출처>)`를 출력해 예산 기준이 probe 줄, sysfs, host memory 중 어디서 왔는지 밝힙니다.
  - *순서.* backend, 하드웨어, 예산 감지는 스크립트 로드 시점에서 인자 파싱 이후 호출되는 `resolve_platform`으로 옮겨졌습니다. probe에 `MODELS_DIR`과 모델 인자가 필요하기 때문입니다.
  - *컬럼.* 모든 행의 끝 커밋 필드는 이제 `COMMIT_FIELDS`에서 옵니다. ROCm에서는 `src/lib/mlx-cpp/patches-rocm/UPSTREAM`에서 읽은 8자리 overlay 커밋과 HIP 버전이 붙고, 헤더에 `,mlx_rocm_overlay_commit,hip_version`이 추가됩니다. Metal과 CUDA 행은 이전 스키마를 유지합니다. 장치 이름과 버전 문자열은 따옴표 없는 CSV 필드에 들어가기 전에 쉼표와 따옴표가 제거됩니다.
- **`scripts/bench_mlxlm.py`** (107줄 변경). `detect_rocm_gpu`는 동작하는 `nvidia-smi`가 있으면 물러나고, 없으면 `rocminfo`의 첫 GPU agent, ROCm 버전 파일, sysfs VRAM을 파싱합니다. `rocm_hardware_names`는 셸 스크립트와 같은 short/full 태그를 만듭니다. ROCm에서는 `PYLM_BENCH_MAX_GB`가 없을 때 메모리 한도가 device memory의 85%가 되고, `baseline_version`에 `+mlx-<mx.__version__>`이 붙습니다. 그곳의 MLX는 source build이고 버전 문자열에 커밋이 들어 있기 때문입니다.
- **`scripts/compare_bench_csv.py`** (25줄 변경). `runtime_and_host`는 하니스 파일명의 첫 필드가 `metal`, `cuda`, `rocm`, `pylm` 중 하나일 때 `_`로 나눈 앞 두 필드를 취합니다. `newer_readings_elsewhere`는 하드코딩된 호스트 목록 대신 이것을 씁니다.
- **`tests/test_bench_rocm_detection.py`** (신규, 351줄). 셸 함수를 추출해 따로 실행합니다. probe 파서, `rocminfo` agent 선택, 버전 도구 부재, 하드웨어 태그(새 AMD 태그와 기존 Apple, GB10, V100 태그), stub `mlxcel`에 대한 `probe_runtime`(가장 작은 체크포인트 선택, 파싱된 장치 필드, ROCm이 아닌 backend, `rocminfo` 대체 경로), backend와 예산 선택, Python 하니스가 같은 태그를 만드는지, ROCm 호스트에서의 비교 스캔을 다룹니다. PR 본문은 태그와 비교 테스트가 변경 전 스크립트에서 실패함을 확인했다고 적고 있습니다.
- **`benchmarks/rocm_strixhalo-gfx1151_2026-09-30.csv`**, **`benchmarks/pylm_strixhalo-gfx1151_2026-09-30.csv`**(신규, 각 5행)와 **`docs/benchmark_results/rocm-baseline-gfx1151-2026-09-30.md`**(신규): 기준선과 그 설명.
- **`docs/benchmarks.md`**는 새 컬럼 두 개, `baseline_version` 접미사, 명령어가 담긴 "ROCm hosts (issue #1810)" 절을 문서화합니다. **`docs/installation.md`**는 ROCm 절에서 기준선으로 링크합니다.

커밋 이력: `9d412eef`는 감지, 표시, 예산, 컬럼을 추가하고, `5fe7e13f`는 이를 문서화하고, `fb1ee3e8`은 기준선을 공개합니다. `a374a2d6`은 ROCm 도구나 파일이 없을 때 helper가 스크립트를 조용히 종료시키지 않도록 하고, `c9c3b92c`는 가장 작은 체크포인트로 probe하고 `rocminfo`에 시간 제한을 두며 CSV 필드를 정리합니다. `d909b974`는 `probe_runtime` 테스트를 추가합니다.

## 3. 기술적 선택과 그 이유

### CLI 플래그 대신 1토큰 실행으로 probe

이슈는 probe 실행과, 모델 로드 없이 `gpu_backend_kind()`를 출력하는 새 플래그 중 probe를 몇 초 안에 끝낼 수 있는 쪽을 택하라고 했습니다. probe는 Rust 변경이 필요 없고, `rocm_smoke.sh`가 이미 확인하는 줄을 그대로 읽으며, 바이너리가 무엇으로 컴파일되었는지가 아니라 장치에서 실제로 무엇이 돌았는지를 보고합니다. Strix Halo 호스트에서 Qwen3-0.6B-4bit로 약 4초가 걸립니다. 대가는 probe에 로드 가능한 체크포인트가 필요하다는 점이고, 그래서 감지가 인자 파싱 뒤로 옮겨졌고 대체 경로가 존재합니다.

### 지정된 모델이 아니라 가장 작은 체크포인트로 probe

첫 버전은 명령줄에 지정된 모델로 probe했습니다. 리뷰에서 probe가 `bench_one`의 `SKIP:oom_estimate` 가드보다 먼저 실행된다는 점이 드러났습니다. 예산보다 큰 체크포인트로 단일 모델 실행을 하면 그래도 로드되고, 큰 모델은 두 번 로드됩니다. 이제 probe는 저장소에서 가장 작은 체크포인트를 쓰고, 저장소가 비어 있을 때만 지정된 모델을 씁니다.

### "backend 줄 없음"과 "다른 backend"를 구분

아무것도 출력하기 전에 죽은 probe는 호스트에 대해 아무것도 말해 주지 않으므로 `rocminfo`가 결정합니다. `metal`이나 `cuda`를 출력한 probe는 적극적인 증거이므로, `rocminfo`가 AMD agent를 나열하더라도 그쪽이 우선합니다. 깨진 체크포인트가 sweep 표시를 잘못 만들지 않게 하면서, ROCm이 아닌 바이너리를 ROCm으로 바꿔 부르는 일도 없게 합니다.

### ROCm에서만 device memory를 예산 기준으로

UMA APU에서 GPU가 할당할 수 있는 양은 carve-out이며, `mlxcel generate`의 자체 preflight도 PR #1883 이후 ROCm allocator의 `memory_limit()`을 읽습니다. 그 밖의 모든 곳, 그리고 ROCm에서 장치 수치를 찾지 못한 경우에는 host memory가 기준이며, 이때 스크립트는 조용히 쓰지 않고 host memory로 대체되었다고 출력합니다.

### 새 컬럼은 ROCm 행에만

`mlx_rocm_overlay_commit`과 `hip_version`을 모든 백엔드에 붙이면 항상 비어 있을 컬럼 때문에 Metal과 CUDA 스키마가 바뀌었을 것입니다. `compare_bench_csv.py`는 `csv.DictReader`로 헤더 이름을 기준으로 행을 읽으므로, 한 런타임 파일에만 있는 끝 컬럼이 짝짓기를 방해하지 않습니다. `mlx_commit`은 이슈가 요구한 대로 upstream pin의 의미를 유지합니다.

### 파일명에서 runtime과 host를 파싱

두 하니스 모두 이미 파일명을 `<runtime>_<host>_...`로 짓습니다. 앞 두 필드를 읽으면 유지할 목록 없이 superseded-baseline 스캔이 모든 호스트로 일반화됩니다. PR 본문이 기록한 부수 효과가 하나 있습니다. 예전 코드가 건너뛰던 `cuda_*` 파일에서도 이제 스캔이 돕니다.

### 경합 가드 아래에서 기준선 공개

이 호스트는 다른 GPU 작업도 돌립니다. 페이지는 KFD 프로세스와 컴파일러가 없는 상태가 90초 이어질 때까지 기다리고, sweep 동안 둘을 초당 한 번 샘플링하며, 창 안에서 둘 중 하나라도 보인 sweep은 버리는 래퍼를 설명합니다. 다섯 번의 sweep이 버려지고 다시 실행되었습니다. 공개된 mlxcel sweep은 KST 02:31:32부터 02:40:28까지, mlx-lm sweep은 KST 03:21:29부터 03:29:42까지 실행되었습니다.

## 4. 기준선 결과

커밋된 CSV 기준(pp512/tg128, batch 1, greedy, 20토큰 warmup 뒤 측정 1회, 모델마다 프로세스 하나):

| 모델 | mlxcel prefill tok/s | mlx-lm prefill tok/s | mlxcel decode tok/s | mlx-lm decode tok/s | Decode 비율 |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B-4bit | 4417.27 | 4316.74 | 278.48 | 220.13 | 1.27x |
| Meta-Llama-3.1-8B-Instruct-4bit | 1065.78 | 966.84 | 35.41 | 32.81 | 1.08x |
| Qwen3-30B-A3B-4bit | 275.65 | 280.40 | 61.82 | 58.65 | 1.05x |
| gpt-oss-20b-MXFP4-Q4 | 7.69 | 7.77 | 8.07 | 8.07 | 1.00x |
| Mixtral-8x7B-Instruct-v0.1-4bit | 25.85 | 26.43 | 7.65 | 9.98 | 주장하지 않음 |

두 파일의 하드웨어 문자열: `AMD_Radeon_8060S_Graphics_gfx1151_ROCm10.0.0_96GB`. mlxcel 행: `mlxcel_commit` `2fcdbe5f`, `mlx_commit` `81ba1c6a`, `mlx_rocm_overlay_commit` `75915908`, `hip_version` `7.15.26333`. mlx-lm 행: `baseline_version` `mlx-lm-0.31.3+mlx-0.32.3.dev20260912+a7d4c85a`. 페이지는 `2fcdbe5f`가 빌드 커밋 `784e35b7`와 하니스 스크립트, 그 테스트, 문서만 다르므로 바이너리는 같다고 기록합니다.

페이지의 해석:

- dense 또는 affine MoE 모델 세 개에서 mlxcel의 decode는 8B와 30B-A3B에서 5에서 8%, 토큰당 호스트 오버헤드 비중이 가장 큰 0.6B에서 27% 앞섭니다. Prefill은 어느 쪽으로든 11% 이내입니다.
- expert가 범용 `gather_qmm` 경로를 거치는 MoE 체크포인트 두 개는 두 런타임에서 똑같이 느립니다. Mixtral prefill은 약 26 tok/s, gpt-oss는 8 tok/s 미만(512토큰에 약 66초)입니다. 두 런타임이 3% 이내로 일치하므로 원인은 mlxcel의 모델 코드가 아니라 overlay의 공유 커널입니다. fused MoE 커널은 아직 ROCm 포트가 없습니다(#1814).
- `compare_bench_csv.py --reference`는 다섯 행을 모두 빠짐없이 짝짓습니다.
- 모든 모델이 96 GiB의 85% 안에 들어갔고, 건너뛴 모델도 실패한 실행도 없습니다.

## 5. 검증

PR 작성자의 실행(gfx1151, ROCm 10.0.0, HIP 7.15.26333, `784e35b7`에서 `--features rocm`으로 빌드한 release 바이너리): 단위 테스트(35개 통과), `make verify-*` 게이트, `insert_apache_header.py --check`, `dead_doc_pointers` 테스트가 통과했습니다. 종단 간 `bench_decode.sh all`은 `rocm`, `strixhalo-gfx1151`, 96 GiB 예산을 감지하고 다섯 모델을 모두 측정했으며, `bench_mlxlm.py all`은 짝이 맞는 `pylm_strixhalo-gfx1151` 파일을 썼습니다. 리뷰 수정 뒤에도 단일 모델 실행은 같은 표시와 예산을 보고했고, probe 체크포인트를 로드할 수 없는 실행은 sysfs에서 device memory를 가져왔습니다. 이 실행들은 GPU를 공유했으므로 수치는 공개하지 않았습니다.

오케스트레이터 검증(gfx1151 호스트, origin/main `81f13ecd` 위로 rebase한 브랜치). 변경이 스크립트, 문서, CSV, Python 테스트뿐이므로 전체 테스트 대신 영향받는 게이트를 실행했습니다.

- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: 통과.
- `make verify-binary-assets`: `tests/fixtures/nemotron_parse_page.png`(41.1 KB, 32 KB 한도 초과)에서 실패. main에서도 똑같이 실패하며(#2034가 추가한 fixture), 이 PR과 무관합니다.
- `bash -n scripts/bench_decode.sh`와 `scripts/bench_mlxlm.py`, `scripts/compare_bench_csv.py`의 `py_compile`: 통과.
- `pytest tests/test_bench_rocm_detection.py tests/test_bench_decode_oom_classifier.py`: 35개 통과.
- `cargo test --release --features rocm --test dead_doc_pointers`: 2개 통과.

## 6. 학습 포인트

- **권위 있는 정보가 바이너리 안에만 있다면 짧은 실제 실행도 올바른 probe입니다.** `gpu_backend_kind()`를 보여 주는 플래그는 없지만, 1토큰 `generate`가 출력하는 줄은 결정된 backend, target, device memory를 알려 줍니다. CI smoke 테스트가 이미 확인하는 줄을 재사용하면 하니스와 CI가 같은 계약을 읽게 됩니다.
- **`set -euo pipefail` 아래 command substitution에서 도는 감지 코드는 절대 실패하면 안 됩니다.** ROCm 버전 파일이 없거나 `hipconfig`가 실패하면 첫 버전의 스크립트는 메시지 없이 끝났습니다. 이제 모든 helper가 빈 필드를 반환하고, 도구가 없는 상태에서 이를 실행하는 테스트가 있습니다.
- **짝지어야 하는 두 하니스는 서로 다른 출처에서 같은 태그를 만들어야 합니다.** 셸 하니스는 장치 이름을 probe의 `GPU:` 줄에서, Python 하니스는 `rocminfo`의 marketing name에서 읽고, 메모리는 probe 또는 sysfs에서 옵니다. 이 호스트에서는 둘이 일치하고(둘 다 `AMD_Radeon_8060S_Graphics_gfx1151_ROCm10.0.0_96GB`), 같은 입력에 대해 Python 태그를 셸 태그에 고정하는 테스트가 있습니다. probe 이름과 marketing name이 다른 호스트라면 짝이 갈라질 것입니다.
- **공유 GPU에서의 벤치마크에는 장담이 아니라 기록된 가드가 필요합니다.** 다른 유닛의 GPU 테스트가 겹친 경우를 포함해 버려진 sweep들이 가드의 필요성을 보여 줍니다. 가드의 창을 기록해 두면 독자가 수치를 판단할 수 있습니다.
- **하드코딩된 호스트 목록은 조용한 꺼짐 스위치입니다.** superseded-baseline 검사는 작성된 이래 Apple이 아닌 모든 호스트에서 꺼져 있었습니다. 파일명 규칙에서 호스트를 끌어내면 목록이 필요 없어집니다.

## 7. 주의 사항과 검증되지 않은 부분

- **mlx-lm 쪽은 mlxcel의 MLX와 바이트 단위로 같지 않습니다.** MLX에는 ROCm wheel이 없으므로 Python 기준선은 spike 트리의 source build(MLX `a7d4c85a`)에서 돕니다. 같은 fork 커밋(`75915908`)을 같은 MLX pin(`81ba1c6a`) 위에 올린 것이고, `LOCAL_FIXES.md` 항목 1에서 6, 8에서 11에 해당하는 시험 커밋과 항목 7의 일부를 갖지만, 항목 12에서 19(f16 `gather_qmm` 빠른 경로, tiled qmv의 f32 activation, scatter 인자 폭, `SearchSorted`, `Hadamard`, 좁은 gather index, FFT, `get_launch_args` 제거)는 없습니다. 페이지는 이것들이 bf16 텍스트 decode에 영향을 주지 않는다고 판단하지만, 실행된 커널을 추적해 확인하지는 않았습니다. mlxcel 자체 overlay 소스로 빌드한 기준선이 있으면 이 문제가 해소됩니다.
- **Mixtral decode는 재현되지 않습니다.** 깨끗한 측정값은 mlxcel 7.65, 8.22, 11.16 tok/s, mlx-lm 9.98, 8.11이었고, 같은 실행의 prefill은 25.8에서 26.5에 머물렀습니다. sweep의 0.77x는 잡음이며 비율을 주장하지 않습니다. 원인(클럭 또는 전력 상태, 가중치 배치, 또는 다른 무엇)은 #1814로 넘겨졌습니다.
- **MoE prefill은 두 런타임 모두에서 느립니다.** Mixtral과 gpt-oss는 expert를 공유된 범용 `gather_qmm`으로 돌립니다. 이는 #1814의 출발점이지 mlxcel의 회귀가 아닙니다.
- **모델당 측정 1회.** Mixtral 반복 측정을 빼면 각 수치는 한 번의 실행입니다. 나머지 네 모델의 실행 간 편차는 측정하지 않았습니다.
- **Metal과 CUDA 호스트.** 이 환경에서는 쓸 수 없습니다. 해당 경로의 변경은 감지를 인자 파싱 뒤로 옮긴 것과 `${COMMIT_FIELDS}` 치환입니다. 단위 테스트가 Apple, GB10, V100 태그와 host memory 경로를 고정하지만 Metal이나 CUDA sweep은 실행하지 않았습니다. `compare_bench_csv.py`는 이제 예전에 건너뛰던 `cuda_*` 파일에서도 superseded 측정값을 찾으므로, 이전에는 조용하던 CUDA 비교에서 경고가 나올 수 있습니다.
- **다른 `gfx` target.** gfx1151만 실행했습니다. 외장 AMD GPU에서의 `amd-<gfx>` 태그와 `rocminfo` 대체 경로는 단위 테스트로만 다뤄집니다.

## 8. 남은 작업

- #1814(ROCm 성능)는 이 기준선을 기준으로 측정합니다. 페이지는 MoE prefill 격차(fused MoE 포트 없음)와 Mixtral decode 편차의 원인을 그 에픽으로 넘깁니다.
- mlx-lm 쪽을 mlxcel 자체 overlay 소스로 다시 빌드하면 빌드 차이에 대한 주의 사항이 사라집니다.

참고: #1810(이 PR로 닫힘), #1801, #1802, #1805, #1808, #1809, #1814, #2034, PR #1818, PR #1883.
