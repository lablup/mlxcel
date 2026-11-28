# 기술 보고서: PR #2018, ROCm에서 남은 커스텀 커널 런처 네 곳에 가드 추가

**날짜**: 2026-09-28

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기. Metal과 CUDA는 실행하지 않았습니다. 여기에 없는 하드웨어이며, 영향이 없다는 근거는 측정이 아니라 도달 가능성 논증이고 아래에 그대로 밝혔습니다.

**언어**: C++ (HIP/CUDA/Metal 디스패치), Rust (cxx 브리지와 호출자)

**위험도**: 중간. 프로덕션 모델 파일 일곱 개에 닿지만, 새로 생긴 경로는 Metal과 CUDA에서 전부 도달 불가입니다.

## 요약

mlxcel은 융합 커널마다 Metal 포트와 CUDA 포트 사이에서 디스패치합니다. 이슈 #1803 이전에는 그 판단이 `!metal::is_available()`이었고 "CUDA"로 읽혔습니다. #1803이 이름 있는 백엔드 종류로 바꿨지만 **false 쪽 팔은 여전히 "Metal"을 뜻한 채** 남았습니다. ROCm에서는 그 팔을 타고 `fast::metal_kernel`이 던지며, 브리지 선언이 `Result`가 아니라서 그 throw가 `noexcept` cxx extern을 가로질러 프로세스를 끝냅니다. 이슈 #1885가 샘플러 런처 둘을 고쳤고, 이 PR이 남은 넷을 닫습니다. `fused_add_rms_norm`, `fused_rope_qk_append`, `ssm_update_kernel`, `run_fused_moe_two_kernel`입니다.

이 변경은 포팅이 아니라 거부입니다. 여기 추가한 가드는 전부 "이 백엔드에는 커널 포트가 없다"는 표지판이고, 포트를 공급하는 것은 이슈 #1814입니다.

## 문제 정의

프로세스 abort는 오류보다 나쁜 실패입니다. 이유가 셋입니다. 요청 하나가 아니라 서버 전체를 내리고, 호출자가 분기할 수 있는 타입 값을 남기지 않으며, 메시지가 이유가 아니라 **시도해본 포트 이름**을 말합니다. 마지막 것은 적극적으로 오도합니다. AMD 호스트에서 `[metal_kernel] No Metal back-end.`는 Metal 설치가 없다는 뜻으로 읽히지만, 실제 사실은 ROCm GPU가 멀쩡히 있고 그 커널의 포트만 없다는 것입니다.

이 결함은 눈으로 찾기도 어렵습니다. 런처 위의 라우팅 층은 정상이기 때문입니다. 프로덕션은 지원 술어로 게이트하고 그래프 폴백을 타므로 평범한 서빙에서는 나쁜 팔에 닿지 않습니다. 닿는 것은 **직접 호출**입니다. 테스트, 벤치마크, 그리고 술어를 건너뛰는 모든 호출자입니다.

## 변경 요약

같은 형태의 디스패치 지점이 열 곳 있습니다. 눈으로 읽는 대신 각 지점 앞 40줄에 `custom_kernels_available()` 가드가 있는지 기계적으로 검사해 분류했습니다. 놓치지 않기 위해서입니다.

| 이미 가드됨 | 무방비, 이번에 수정 |
|---|---|
| `gumbel_max_sample` (#1885) | `fused_add_rms_norm` |
| `rejection_sample` (#1885) | `fused_rope_qk_append` |
| `paged_attention_decode` (#1803) | `ssm_update_kernel` |
| `paged_attention_decode_v2_partial` | `run_fused_moe_two_kernel` |
| `paged_attention_merge_states` | |

- `src/lib/mlx-cpp/turbo/fused_norm.cpp`, `fused_rope_append.cpp`, `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`: 포트를 고르기 **전에** `custom_kernels_available()` 가드를 두고, 진입점 이름·누락된 포트·호출자가 대신 타야 할 폴백을 메시지에 담아 던집니다.
- `src/lib/mlxcel-core/src/lib.rs`: 브리지 선언 다섯 개가 `Result`가 됩니다. 넷이 아니라 다섯인 이유는 `run_fused_moe_two_kernel`이 `fused_moe_expert_kernel`과 `fused_moe_geglu_kernel` 두 진입점으로 도달하기 때문입니다.
- 호출 지점 열 곳을 `switch_layers.rs`, `gemma4.rs`, `qwen3_next.rs`, `falcon_h1.rs`, `nemotron_h.rs`, `granitemoehybrid.rs`, `plamo2.rs`, `layers.rs`, `fused_moe_parity_tests.rs`에서 갱신했습니다.
- 이 변경이 거짓으로 만든 주석 두 개를 고쳤습니다.

## 기술적 선택과 그 이유

### 핵심은 가드가 아니라 `Result` 전환입니다

선언을 바꾸지 않고 가드만 넣으면 abort 메시지 하나가 다른 abort 메시지로 바뀔 뿐입니다. `noexcept` extern을 가로지르는 C++ throw는 메시지와 무관하게 `std::terminate`입니다. 두 반쪽은 함께 들어가야 하며, 앞으로 런처 가드를 추가할 때도 선언 변경이 빠지면 미완성입니다.

### 가드를 넣기 전에 각각의 프로덕션 게이트를 확인했습니다

프로덕션이 실제로 지나는 경로 위에 가드를 놓으면, 잘 돌던 그래프 폴백이 throw로 바뀝니다. 각각 확인했습니다.

- `fused_add_rms_norm_available()`과 `fused_rope_qk_append_available()`는 곧 `mlxcel::custom_kernels_available()`입니다.
- `ssm_kernel_available()`는 non-Apple에서 `cu::is_available()`을 반환하는데, ROCm 빌드에서는 no-CUDA 스텁이라 false입니다.
- `fused_moe_enabled()`는 환경변수 검사에 `custom_kernels_available()`를 접어 넣습니다.

넷 다 ROCm에서 false이므로 프로덕션은 그래프 경로를 타고 어떤 가드에도 도달하지 않습니다. 게이트에 회귀가 없는 이유도 이것입니다. 가드가 걸린 경로는 원래 지나지 않던 길입니다.

### 호출자가 세 갈래로 갈리며, 어느 쪽으로든 통일하면 결함이 됩니다

MoE 지점들은 `.ok()`입니다. 감싸는 함수가 `Option`을 반환하고 지원하지 않는 설정마다 이미 `None`을 돌려주므로, "이 백엔드에 포트가 없다"는 그 계약의 한 사례이고 호출자는 SwitchGLU 그래프 경로로 넘어갑니다. 포트 없는 백엔드에서의 직접 호출은 결함이 아니라 정당한 사용이라, 여기에 `expect`를 쓰면 올바른 코드에서 패닉합니다.

SSM·`layers.rs`·패리티 테스트 지점들은 `expect`입니다. 각각 이미 지원 술어로 게이트하므로, 그 자리에서 오류가 나오면 게이트와 런처가 서로 다른 말을 한다는 뜻입니다. 그것을 조용한 폴백으로 접으면, 이 작업이 신뢰할 수 있게 만들려는 바로 그 층의 실제 버그를 감춥니다.

한 관용구로 통일했다면 맞지 않는 쪽에서 결함이 생겼을 것입니다.

### 낡아진 주석 둘을 남기지 않고 고쳤습니다

`fused_moe_enabled()`는 포트 검사 항을 이렇게 정당화하고 있었습니다. "브리지 함수가 `Result`를 반환하지 않으므로, 포트 없는 백엔드에서 `fast::cuda_kernel` throw가 프로세스를 끝낸다." #1803 시점에는 정확했고 이 변경 이후로는 거짓입니다. 항 자체는 남기되, 더 약한 새 이유를 그 자리에 적었습니다. 평탄화한 인자를 다 만들어놓고 거부당하는 것보다 미리 판단하는 편이 싸다는 것입니다.

`fused_add_rms_norm_eligible()`은 런처의 예외가 복구 불가능하다고 주장했습니다. 모양 불일치 같은 계약 위반에는 여전히 참이지만 포트 부재에는 더 이상 참이 아닙니다. 이제 두 부류를 구분해 적습니다.

존재하지 않는 제약을 설명하는 주석은 없느니만 못합니다. 다음 사람이 그 제약을 믿고 설계하기 때문입니다.

## 검증

- gfx1151에서 전체 ROCm 게이트, `cargo test --workspace --profile test-fast --features rocm --no-fail-fast -- --test-threads=1`: 실패 타깃 하나 `-p mlxcel-core --lib`로 이 브랜치 이전과 동일합니다. 그 실패는 이슈 #1806의 nvfp4 abort이며, 실행 전체에서 유일한 `terminate called`입니다.
- `cargo fmt --all -- --check`: 깨끗합니다.
- `cargo clippy -p mlxcel --lib --tests -- -D warnings`, CI가 돌리는 바로 그 명령: 깨끗합니다.
- 거부는 테스트 통과에서 추론하지 않고 **실행해서** 확인했습니다. ROCm 빌드에서 일회용 프로브가 `[fused_add_rms_norm] no custom kernel port for this GPU backend; mlxcel's callers take the graph fallback instead`를 타입 있는 `Err`로 반환했고 프로세스는 살아 있었습니다.

### 남겨둘 만한 방법론

열 곳 중 둘은 타입 오류로 나타나지 않습니다. `Result<()>`를 무시하는 것은 타입 불일치가 아니라 `unused_must_use` 린트라서, `cargo check`는 통과했고 `-D warnings` clippy에서만 드러났습니다. 둘 다 `layers.rs`의 디코드 핫패스 호출자입니다. `Result<T>` 전환은 컴파일러가 잡아주지만 `Result<()>` 전환은 린트가 필요하며, `cargo check`에서 멈추는 작업 흐름은 그 구멍을 그대로 내보냅니다.

## 검증하지 않은 것

- **Metal과 CUDA를 실행하지 않았습니다.** 이 호스트에 없습니다. 영향이 없다는 주장은 그곳에서 `custom_kernels_available()`가 true라 새 분기가 전부 도달 불가라는 점과, 성공 경로가 그대로라는 점에 기댑니다. 측정이 아니라 논증입니다.
- **네 거부 중 하나만 프로브했습니다.** `fused_add_rms_norm`은 실행으로 확인했습니다. 나머지 셋은 동일한 가드 형태와 선언 변경을 공유하지만, 호출해본 것이 아니라 컴파일과 회귀 없는 게이트로 확인했습니다.
- **MoE의 `.ok()` 경로는 게이트가 지나지 않습니다.** ROCm에서는 `fused_moe_enabled()`가 이미 false라 프로덕션이 커널을 부르지 않고 새 `.ok()`에 닿지 않습니다. 그 분기는 테스트가 아니라 읽기로 덮인 상태입니다.

## 남은 작업

이슈 #1814가 이 가드들이 대신 서 있는 포트를 공급합니다. 길은 추측이 아니라 이미 열려 있습니다. `fast::hip_kernel`이 ROCm 오버레이의 `src/lib/mlx-cpp/patches-rocm/mlx/fast.h:96`에 있고, 이슈 #1862가 BitNet 커널로 3-암 스위치를 끝까지 증명했습니다.

조사 중 확정된 사실 둘을 #1814로 가져갈 만합니다. 첫째, `fast::hip_kernel`은 런타임에 hipRTC로 컴파일합니다(`patches-rocm/mlx/backend/rocm/jit_module.cpp`). Metal·CUDA와 같은 모델이므로, 이슈 #1853에서 dtype별 인스턴스화를 비싸게 만든 사전 컴파일 비용은 오버레이의 `.hip` 파일에 적용되고 mlxcel 자체 융합 커널에는 적용되지 않습니다. 둘째, 실제 제약은 wavefront 폭입니다. gfx1151은 wave32이고 CDNA 계열은 wave64인데, 16에서 시작하는 리덕션 fold는 wave64에서 조용히 절반만 접고 **유한하고 그럴듯한 틀린 값**을 냅니다. 그리고 `static_assert(warpSize == 32)`는 HIP에서 컴파일되지 않습니다. `warpSize`가 `operator int()`를 가진 객체이기 때문입니다. 이슈 #1862는 전처리기 `#error`로 막았고, 포팅하는 모든 리덕션 커널에 같은 처리가 필요합니다.
