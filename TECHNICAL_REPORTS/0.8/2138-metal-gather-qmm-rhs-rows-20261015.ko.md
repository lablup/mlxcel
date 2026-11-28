# PR #2138: Metal `gather_qmm_rhs`의 행 수를 broadcast된 x 기준으로 지정

**작성일**: 2026-10-15
**상태**: #2304 이후 main `9bfabbab`과 통합. 남은 변경은 출처 표시와 보고서이며 프로덕션 소스는 검증된 main과 동일
**위험도**: 낮음 (모든 프로덕션 호출에서 no-op이며, `gather_qmm_rhs`가 x를 broadcast해야 하는 경우에만 동작이 바뀜)

## 요약

M1 Ultra 러너의 Metal nightly는 2026-09-30부터 `models::switch_layers::mxfp_tests::{mxfp4,mxfp8}_gather_qmm_matches_host_reference`의 `SortedSharedActivation` 케이스에서 실패했다. MLX Metal 백엔드의 `GatherQMM::eval_gpu`는 `gather_qmm_rhs`에 행 수로 `x.size() / K`를 넘긴다. 이 값은 `gather_qmm_rhs`가 x를 indices에 맞춰 broadcast하기 전의 행 수다. 활성 행 하나를 정렬된 슬롯 32개에 gather하면 이 값은 1이 되어, grid와 커널의 `M` 경계가 한 행만 덮고 출력의 1..31행은 기록되지 않는다. 이 PR은 overlay에서 `B * M`을 넘긴다. 또한 macOS 경로 정규화와 유효하지 않은 바이트 이름을 거부하는 파일시스템에서 split-MTP 테스트 두 개가 이식성 있게 동작하도록 고친다. 프로덕션 resolver와 심볼릭 링크 보호는 변경하지 않는다.

Refs #1599.

## 1. Nightly 이력

| 날짜 | 실패 | 상태 |
|---|---|---|
| 2026-09-28 | `verify-clippy`: `fused_norm_parity_tests.rs:158`, `fused_rope_parity_tests.rs:133`의 미사용 `Result` | #2029 (d8d34e2b)에서 수정 |
| 2026-09-29 | `tests::family_order_is_exhaustive`: `FAMILY_ORDER`에 `Speech` 누락 | #2080 (5486e404)에서 수정 |
| 2026-09-30 ~ 10-11 | mxfp gather 테스트 두 개 | 이 PR에서 수정; `25202198`에서 집중 Metal 검증 통과 |
| 2026-10-12 (`1e8c7022`) | 같은 mxfp 테스트, split-MTP의 `/var` 정규화 및 유효하지 않은 바이트 파일명 가정, YAML `err_expect` lint 세 개 | 커널과 split-MTP 테스트는 이 PR에서 수정; YAML lint는 현재 main에서 이미 수정됨 |

mxfp 테스트는 2026-09-30 #2071 (2cee9cf4)과 함께 추가되었으므로, 마지막 실패는 회귀가 아니라 새 테스트가 upstream의 잠재된 경계 사례를 드러낸 것이다.

## 2. 근본 원인

- 케이스 입력은 x `[1, 1, K]`, expert 8개에 대한 indices `[32]`, `sorted = true`다. M == 1, B = 32, B / E = 4, `right_sorted_`이므로 `eval_gpu`는 `gather_qmm_rhs`로 간다.
- `gather_qmm_rhs`는 x를 32행으로 broadcast하지만(`broadcast_with_indices`), `M` 인자는 호출자가 원래 x로 계산한 1이다. `grid_dims.y = ceil(M / 16)`과 커널의 `tgp_bm = min(BM, M - y_row)` 때문에 0행만 기록된다. `out`은 `allocator::malloc`에서 오므로 나머지 행에는 이전 메모리 값이 남는다: mxfp4는 상대 오차 7.4e32, mxfp8은 non-finite.
- 같은 실행에서 앞선 gather 케이스 세 개는 bf16, f16, f32 모두 통과했고, MLX CPU 백엔드에서 `SortedSharedActivation`을 돌리는 `mxfp_matmuls_match_host_reference_on_cpu_device`도 통과했다. reference와 테스트는 올바르며, 메시지의 "bf16"은 처음 시도한 dtype일 뿐이다.
- 결함은 모드와 무관하고(affine도 동일), ml-explore/mlx main에도 그대로 있다.

## 3. 변경

유일한 호출 지점에서 `x.size() / K`를 `B * M`(출력 행 수)으로 바꿨다. x가 이미 확장된 경우 기존 값과 같으며, 모든 프로덕션 호출(`SwitchLinear::forward(.., true)`는 항상 `gather_sort` 출력을 받음)이 이에 해당한다. 같은 인자를 받는 NAX 변형도 함께 고쳐진다. overlay 헤더와 CMake overlay 주석에 새 hunk를 기록했다.

`.` 및 `..` 출력의 split-MTP 테스트는 staging 부모를 정규화된 임시 디렉터리 경로와 비교한다. 이는 macOS에서 `/var`가 `/private/var`로 해석되는 production resolver의 의도와 일치한다. 비 UTF-8 테스트는 임시 파일시스템의 바이트 이름 지원을 먼저 검사한다. 지원하는 파일시스템에서는 성공 round trip을 그대로 검증하고, 거부하는 파일시스템에서는 같은 I/O 오류가 반환되며 출력이나 staging 항목이 남지 않아야 한다.

현재 main을 일반 merge하면서 Xcode 27 NAX 호환 파일을 `src/lib/mlxcelverse/overlays/metal/`로 옮겼다. 이 overlay는 MLX 전체 파일의 파생물이므로 기존 Apple 헤더를 유지하면서 MLX 원본 URL, MIT 라이선스 문구, 최상위 `NOTICE` 포인터를 추가했다. `NOTICE`는 전체 overlay를 범위에 포함하고 MLX 저작권과 MIT 조건을 싣고 있으며, README acknowledgment도 라이선스에 따른 재사용임을 밝힌다.

## 4. 검증

- 이전 정확한 head의 Metal gate: Apple M1 Ultra (128 GB), macOS 27.0 (26A428), full Xcode 27 RC, base `b34d517b` 위 PR head `25202198`. Metal overlay를 다시 빌드한 뒤 `models::switch_layers::mxfp_tests` 다섯 개가 모두 통과했고 release 빌드도 통과했다.
- 이전 정확한 head의 모델 스모크: Qwen3-30B-A3B-4bit, greedy 64 tokens에서 base, base 반복, PR이 byte-identical text를 생성했다. SHA-256은 `d2f40b2d5831c3d7790d467fb51123f0d1f615fb7300aa2f92d86a9d8831a0de`다.
- CUDA (GB10): 기존 PR 브랜치가 빌드됐고 `models::switch_layers` 테스트 32개가 모두 통과했다. 이는 Metal 검증이 아니다.
- 현재 main 통합: main `59974283`을 기존 PR 브랜치에 일반 merge하여 `44045522`를 만들었다. 유일한 충돌은 NAX overlay 위치가 `src/lib/mlx-cpp`에서 `src/lib/mlxcelverse`로 옮겨진 것이며, quantized-kernel hunk, split-MTP 테스트, overlay 등록은 의미 충돌 없이 병합됐다. `--profile test-fast --features metal,accelerate`에서 집중 `models::switch_layers::mxfp_tests` 5개가 통과했고 split-MTP 이식성 집중 테스트 2개도 통과했다. formatting, diff, license-header, MLX-pin parser, cross-repository-reference 검사도 통과했다.
- 새 release 실제 모델 A/B: 호환 baseline은 `44045522`에서 `quantized.cpp`만 main `59974283` 버전으로 되돌려 빌드했다. 따라서 `B * M` 변경은 제외하되 Xcode 27 NAX 호환 overlay와 나머지 소스는 동일하다. 이는 pristine main artifact가 아니다. SHA-256은 `d76febfc9adb23ca6bc627310994c108cc4e721c26341ef7f49ab092a89db651`이며, 정확한 `44045522` 후보의 SHA-256은 `ee83df3db3cea7159579487718419d4329c7fbeb147655cb1847a6867b13a872`다.
- 두 release artifact를 각각 두 번씩 Qwen3-30B-A3B-4bit에서 `MLXCEL_FUSED_MOE=0`, greedy decoding, chat template 비활성화, 96-token 반복 단어 prompt, 64개 생성 token으로 실행했다. Qwen3는 token마다 128개 expert 중 8개로 route하므로 prefill은 MoE layer마다 약 768 expert-route slot을 지나 64-slot 경계를 충분히 넘는다. 네 실행의 생성 text는 모두 byte-identical이며 962 bytes, SHA-256 `cbf1670d0b554a35c029ec4104ae3ecfc8a46e07296d7e33a5f3b2c214a5fa8f`다. Timing은 달랐으며 성능 주장에 사용하지 않는다.
- 정확한 소스 후보 Metal nightly [38582194425](https://github.com/lablup/mlxcel/actions/runs/38582194425)는 `lablup-macmini01`의 `44045522`에서 통과했다. Formatting, Clippy, 일반 테스트 12,316개(390개 ignored), video 테스트 51개가 실패 없이 통과했다.
- 리뷰에서 #1599 구현의 남은 기능·보안 문제는 발견되지 않았다. 다만 MLX NAX 전체 파일 overlay의 attribution 누락이 high-severity로 발견됐다. 위에서 설명한 헤더, `NOTICE`, README 변경으로 이를 수정했으며 attribution patch는 후속 리뷰와 저장소 license-header 검사를 통과했다.
- PR CI의 `cargo-deny` job은 2026-10-14 공개된 rustls 0.23.45 권고 네 개를 발견했다. 정확한 main `59974283`도 같은 버전을 잠그므로 #1599가 만든 회귀가 아니라 base-wide 선행 조건이다. 이를 waive하지 않으며 issue #2312와 lock-only PR #2313에서 rustls 0.23.46으로 별도 갱신한다.

## 5. 후속 작업

ml-explore/mlx에 이 호출을 보고하고, upstream 수정이 포함된 pin bump 때 이 hunk를 제거한다. rustls 권고 gate를 억제하지 말고 별도 보안 선행 PR을 merge한다. #1599는 merge 후에도 열어 두고 post-merge Metal nightly가 통과한 뒤에만 닫는다.

## 6. #2304 이후 최종 통합

Main `9bfabbab`에는 #2304를 통해 이 PR의 `B * M` 수정과 split-MTP 이식성 테스트가 이미 포함됐다. CMake와 Metal quantized 파일의 충돌은 해당 main 버전을 유지해 해결했다. Upstream MLX `243bdf09`에는 ml-explore/mlx#4594의 Xcode 27 수정이 있으므로, 오래된 `59d600b5e` 전체 파일을 복원하지 않고 중복 NAX overlay와 등록을 제거했다. NOTICE에서는 제거된 overlay 설명만 삭제하고 확장된 MIT 조건과 expm1f 출처 표시는 유지했다.

결과 프로덕션 트리는 main과 동일하다. 정확한 squash 이전 소스 `4befdbc7`의 Metal nightly [38583586205](https://github.com/lablup/mlxcel/actions/runs/38583586205)는 formatting, Clippy, workspace 테스트 12,315개(390개 ignored), FFmpeg video 테스트 51개를 통과했다. 위 release 모델 검증은 원래 소스에 대한 증거이며 새 pin에서 재실행한 결과가 아니다. #2304가 rustls도 0.23.46으로 올렸으므로 위 보안 선행 조건은 해결됐다.

#1599는 merge 후 main Metal nightly가 통과할 때까지 열어 둔다. M5 전용 실행은 아직 검증하지 못했다.
