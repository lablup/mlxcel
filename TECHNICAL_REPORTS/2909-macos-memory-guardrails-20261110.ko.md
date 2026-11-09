# 대형 상주 모델을 위한 macOS 메모리 가드레일

- **PR**: #2909 (open, 미병합)
- **이슈**: #2883 (close 예정)
- **날짜**: 2026-11-10 (생성일; PR이 아직 열려 있음)
- **브랜치**: `feature/issue-2883-macos-memory-guardrails`
- **작성자**: Jeongkyu Shin
- **리뷰어**: 미지정
- **측정 상태**: 이 변경에 대해 soak 테스트와 시간 측정은 실행하지 않았다.

## 요약

기존 적합성 검사는 모델 로드를 전체 RAM과 Metal 권장 working set에 비교했다. 두 값 모두 macOS가 압축이나 스왑 없이 지금 당장 내줄 수 있는 메모리를 알려주지 않는다. 이 PR은 "지금 사용 가능" 값, `max(7 GiB, RAM의 15%)` 예약량, 그리고 이를 사용하는 가드레일을 추가한다. `inspect`와 `--estimate-memory`의 경고, 서버의 프로세스 메모리 메트릭, 선택형 가중치 `mlock`(`MLXCEL_WIRE_WEIGHTS=1`), `qwen4_exp` PLE row cache의 증가 게이트다. 모든 동작은 권고용이거나 opt-in이다. `FITS` 판정은 그대로이고, 로드를 거부하지 않으며, wiring은 기본 꺼짐이다.

## 문제 정의

대형 모델을 올린 macOS에서 유휴 서버의 가중치가 압축되거나 스왑될 수 있다. 버퍼의 GPU wiring이 마지막 command buffer 후 몇 초 뒤에 풀리기 때문이다. 또한 단계적으로 커지는 저장소(row cache, 이후 expert streaming #2882, prompt-cache 스냅샷 #2885)는 "한 단계 더 가져가도 macOS 예약량 아래로 내려가지 않는가"를 물을 방법이 없었다. 전체 RAM은 가용량을 과대평가하고, Metal working set은 GPU 상한일 뿐 빈 메모리에 대한 값이 아니다.

## 변경 요약

- `src/lib/mlxcel-core/src/host_memory.rs` (신규, 409줄): 재사용 가능한 가용량 값, 예약량, 프로세스 footprint, `AvailabilityGate`.
- `src/lib/mlxcel-core/src/weight_wiring.rs` (신규, 250줄): 평가된 가중치의 선택형 청크 단위 `mlock`. `weight_materialize.rs`에서 호출.
- `cpp/mlx_cxx_bridge.{cpp,h}`: 새 `array_data_address` 브리지 호출.
- `src/execution/memory_availability.rs`, `memory_estimate.rs`: `inspect`와 `--estimate-memory`의 macOS 섹션과 경고.
- `src/server/model_worker.rs`, `routes/metrics.rs`, `routes/metrics_process_memory.rs`: 로드 후 footprint 로그와 `/metrics` 게이지.
- `src/models/qwen4_exp_ple_cache.rs`, `qwen4_exp_ple_storage.rs`: row cache가 예약량 전에 증가를 멈춤.
- `scripts/macos_memory_soak.sh`(커밋됨, 미실행), `docs/macos-memory.md`, 환경 변수 문서, CHANGELOG.

### host_memory.rs (diff를 직접 읽고 작성)

- **지금 사용 가능**은 `free + purgeable + min(inactive + speculative, file_backed)` 페이지다. `min` 상한은 다른 프로그램의 inactive 익명 메모리(macOS가 압축이나 스왑으로만 회수하는 메모리)를 제외한다. `file_backed`는 `external_page_count`이며 active 파일 페이지도 포함하므로, 이 상한은 회수 가능 캐시의 정확한 값이 아니라 느슨한 상한이다.
- **speculative 이중 계산 수정**: 커널의 `free_count`에는 이미 speculative 페이지가 들어 있으므로, 플랫폼 코드는 `free_count - speculative_count`(saturating)를 `free`로 넘기고 speculative는 `min` 안에서 한 번만 더한다. 두 번째 커밋이 이 수정이다. 고정 입력 테스트로 검증했고 `vm_stat`과 대조했다.
- **예약량**: `max(7 GiB, RAM의 15%)`, 오버플로를 피하려 `u128`로 계산. `usable_bytes`는 가용량에서 예약량을 뺀 값(0에서 포화). `shortfall_bytes(demand)`는 요구량이 usable을 넘는 만큼이다. `wired_ceiling_bytes`는 전체 RAM에서 예약량을 뺀 값이다.
- **알 수 없음은 0이 아님**: macOS가 아니거나 커널 질의가 실패하면 두 질의 모두 `None`을 반환하고, 호출자는 이를 "알 수 없음"으로 보고 기존 동작을 유지한다.
- **플랫폼 코드**: `sysctlbyname`의 `hw.memsize`(`OnceLock` 캐시), host port를 한 번만 얻는 `host_statistics64`(`mach_host_self` 호출마다 send right가 늘어나므로), `phys_footprint`까지 손으로 선언한 `TaskVmInfoRev1` 구조체에 대한 `task_info(TASK_VM_INFO)`. 커널이 구조체보다 적은 단위를 돌려주면 `None`을 반환하여 일부만 채워진 구조체를 신뢰하지 않는다. 구조체 레이아웃은 `<mach/task_info.h>에 맞춰 수동 선언했고, 헤더와 자동으로 대조하는 검사는 PR에 없다.
- **AvailabilityGate**: 상태를 250 ms 동안 캐시하고, 마지막 조회 이후 승인한 바이트(`granted_since_probe`)를 추적한다. 캐시된 값은 그것을 반영할 수 없기 때문이다. `step + granted_since_probe`가 usable 안에 들어올 때만 승인한다. 거부된 단계는 승인 카운터에 더하지 않는다. 상태를 알 수 없으면 모든 단계를 승인한다. mutex는 poisoning에서 panic 대신 복구한다. probe는 주입 가능해서 테스트에 sleep이 필요 없다.

### weight_wiring.rs (diff를 직접 읽고 작성)

- `MLXCEL_WIRE_WEIGHTS`가 `1`, `true`, `on`, `yes`일 때만 동작한다. `lazy` materialization 정책에서는 평가되지 않은 가중치라 경고만 남기고 아무것도 하지 않는다. macOS가 아니면 경고 후 무시한다.
- 범위(새 브리지 호출의 주소, `nbytes` 길이)는 페이지 단위로 넓혀 정렬하고 병합하므로, 작은 텐서 둘이 공유하는 페이지는 한 번만 잠긴다.
- `wire_ranges`는 커널 호출(`wired_now`, `lock`, `unlock`)을 클로저로 받아 정책을 고정 입력으로 단위 테스트한다. 시작 전에 시스템 wired 합계를 확인하고, 1 GiB 청크씩 잠그며, 청크마다 wired 합계를 다시 읽고, 합계가 `RAM - 예약량`을 넘기는 청크는 풀고 중단한다. `mlock`이 실패하면 해당 범위를 실패로 세고 첫 `errno`를 기록한 뒤 그 범위의 나머지를 포기하고 다음 범위로 넘어간다. 사전 검사는 상한 이상(`>=`)에서, 청크 검사는 초과(`>`)에서 멈추는 작은 비대칭이 있으나 GiB 단위에서는 실질적 영향이 없다.
- 예측이 아니라 실제 시스템 wired 값과 비교한다. Metal이 GPU용으로 이미 wired한 페이지가 그 값에 들어 있고 `mlock`은 그 페이지에 추가하는 것이 없기 때문이다.
- wired 합계를 읽을 수 없으면 상한 도달로 보고 중단한다. 보수적인 방향이다.
- `array_data_address`는 평가 전 배열과 strided view(row도 column도 contiguous가 아님)에 0을 반환한다. strided view는 주소와 `nbytes`가 실제 바이트를 덮지 못하기 때문이다. 이런 텐서는 건너뛰며 요청량에는 잡히지만 wired에는 잡히지 않는다. 세 번째 커밋이 이 수정이다.
- 코드에 명시된 범위 제한: 멀티모달 타워는 lazy로 남아 wiring되지 않고, 첫 토큰의 repack(expert stacking, dtype 변환)은 wired 소스를 wired 아닌 결과로 대체한다.

### 기타

- `qwen4_exp` PLE row cache는 각 단계에 부기 비용을 포함해 청구하고 가용량 조회 하나를 250 ms 재사용한다. prompt-cache 저장소는 의도적으로 게이트하지 않았다. 항목이 삽입 전에 이미 존재하고 바이트 예산이 상한을 두므로 거부할 단계가 없다.
- `/metrics`는 `phys_footprint`, 압축 바이트, MLX active/cache 메모리, wired 가중치 게이지를 내보낸다. macOS가 아니면 프로세스 메모리 계열은 0이 아니라 아예 나타나지 않는다.

## 기술적 선택과 그 이유

- **거부하지 않고 권고만.** `inspect`는 수치와 경고를 출력하고 `FITS`는 그대로다. strict 모드는 실제 압박에 대해 값이 검증되지 않아 제외했다. 배제한 안: 검증되지 않은 휴리스틱으로 로드를 거부하는 것.
- **회수 가능 캐시를 file-backed 페이지로 제한**하여 다른 프로그램의 압축 가능한 익명 메모리를 빈 메모리로 보고하지 않는다. 대가는 상한이 근사치라는 점이다.
- **mlock이 더할 양을 예측하지 않고 실제 wired 값을 읽는다.** Metal 자체의 wired 페이지와 겹치기 때문이다.
- **커널 호출을 클로저로** 받아 macOS 호스트나 권한 없이 wiring과 게이트 정책을 테스트한다.
- **측정 후 유지.** `MLXCEL_WIRE_WEIGHTS`는 꺼짐, `MLXCEL_WIRED_LIMIT`은 기본값, 어떤 로드도 거부하지 않는다. soak로 이점이 확인될 때까지다.
- **공용 게이트**를 `mlxcel_core::host_memory`에 두어 #2882와 #2885가 값을 다시 유도하지 않고 재사용한다.

## 검증과 미검증 사항

수행한 것(PR 본문 기준): 산술, 게이트, wiring 정책, 리포트 문구, 메트릭 단위 테스트, 평가된 MLX 버퍼의 실제 `mlock`, OptiQ 체크포인트의 `mlxcel inspect`(이중 계산 수정 이전이며 512 GB 호스트라 경고가 나오지 않아 경고 경로는 단위 테스트로만 확인), wiring을 켠 `qwen3.5-4b-4bit` 짧은 서버 실행(2.20 GiB wired), 두 라이브러리 crate의 clippy, metal과 accelerate `cargo check`, fmt, 환경 변수 문서 검사. Linux는 컴파일하지 않았고 shellcheck는 설치되어 있지 않다.

**soak 테스트와 시간 측정은 실행하지 않았다.** `scripts/macos_memory_soak.sh`는 커밋만 되었고 실행되지 않았다. 미검증: 128 GB와 64 GB 머신에서 wiring이 유휴 압축을 막는지, 가용량 값이 실제 스왑을 예측하는지, wiring이 디코드나 로드 시간에 미치는 영향, 수동 선언한 `task_vm_info` 레이아웃이 다른 macOS 버전에서도 맞는지, 거부된 청크의 `munlock`이 다른 구성요소가 함께 잠근 페이지와 어떻게 상호작용하는지. `docs/macos-memory.md`에도 측정된 것이 없다고 적혀 있다.

## 학습 포인트

- macOS의 `host_statistics64` `free_count`에는 speculative 페이지가 포함된다. speculative를 따로 더하면 이중 계산이 된다.
- 커널을 다루는 정책은 syscall을 클로저로 주입해 테스트한다.
- "증가 중단" 게이트는 이미 승인한 양의 누계가 필요하다. 캐시된 측정값에는 그것이 보이지 않기 때문이다.

## 후속 작업

- 한가한 128 GB와 64 GB 머신에서 soak를 실행하고 wiring 기본값을 결정한다.
- #2882와 #2885에서 `AvailabilityGate`를 채택한다.
- CI에서 Linux 컴파일로 비macOS stub을 확인한다.
