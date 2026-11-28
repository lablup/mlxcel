# PR #2328: Qwen 3.5 MTP 드래프터 로더에서 양자화 방식 검증

**날짜**: 2026-10-16
**상태**: 구현 완료, 단위 테스트 되돌림 검사와 실제 체크포인트 로드는 머지 전 CUDA(GB10)에서 검증, 엔드투엔드 MTP 추측 생성과 실제 드래프터에서의 5비트 Markov 경로는 미검증 ("미검증 및 후속" 참고)
**위험도**: 낮음 (기존에 abort되던 로드 경로의 검증 단계 추가와, 비트 깊이 목록 하나를 공용 상수로 교체)

## 요약

이슈 #1790은 Qwen 3.5 MTP 드래프터 체크포인트가 지원되지 않는 양자화 방식을 선언할 때 프로세스가 abort되는 문제를 고친다.

1. `Qwen35MtpDraftModel::quantize_dense_projections`(`src/lib/mlxcel-core/src/drafter/qwen3_5_mtp/model.rs`)가 이제 `Result`를 반환한다. 후보 목록이 비어 있지 않으면 `validate_affine_quantization_bits`와 `validate_affine_quantization_group_size`를 호출하고, `from_path`는 오류를 `DrafterError::Config`로 변환한다.
2. DFlash Markov 헤드(`drafter/dflash/markov.rs`)는 비트 깊이 목록을 `SUPPORTED_AFFINE_BITS`에서 가져오므로 5비트 factor가 로드된다.
3. `layers.rs`의 `Used by:` 목록에 새 호출자를 명시한다.

## 문제 정의

`quantize_dense_projections`는 체크포인트 config의 `group_size`와 `bits`를 검사하지 않고 계산에 사용했다. `group_size: 0`이면 `%`에서 panic이 난다. 지원되지 않는 `bits`나 `group_size`는 MLX affine quantize 커널이 `std::invalid_argument`를 던지게 하고, 이는 cxx 브리지를 넘으면서 로드 오류가 아니라 프로세스 abort가 된다. Markov 헤드는 자체 `[2, 3, 4, 6, 8]` 목록을 갖고 있었는데, 커널이 지원하는 5비트가 빠져 있어 유효한 5비트 factor가 거부됐다.

## 기술적 선택과 그 이유

- 후보 목록이 비어 있지 않을 때만 검증한다. 함수가 건너뛰는 이미 변환된 체크포인트는 쓰지 않는 값 때문에 거부되지 않고, `validate_bounds`는 바뀌지 않는다.
- 드래프터 전용 검사를 새로 만들지 않고 공용 affine 검증 함수 두 개를 재사용한다. 허용되는 방식이 `split-mtp` 및 다른 로더와 일치한다. 오류에는 "Qwen 3.5 MTP drafter config" 접두사를 붙인다.
- Markov 헤드의 자체 목록에 5를 추가하지 않고 `SUPPORTED_AFFINE_BITS`로 교체한다. 두 목록이 다시 어긋날 수 없게 하기 위해서다. 설명되지 않는 packed 너비는 여전히 거부한다.
- `quantize_dense_projections`를 `pub(super)`로 바꿔, 명시적 `quantization` 블록이 있는 config로 테스트가 직접 호출하게 한다.

## 변경 요약

- `src/lib/mlxcel-core/src/drafter/qwen3_5_mtp/model.rs`: `Result` 반환, 검증 함수 두 개 호출, `from_path` 변환.
- `src/lib/mlxcel-core/src/drafter/dflash/markov.rs`: 로컬 상수를 `SUPPORTED_AFFINE_BITS`로 교체.
- `src/lib/mlxcel-core/src/layers.rs`: 문서 주석만 변경.
- `src/lib/mlxcel-core/src/drafter/qwen3_5_mtp/tests.rs`: `quantize_dense_projections_rejects_unsupported_quantization`(bits 7, 1, group_size 0, 48, -64), `quantize_dense_projections_only_validates_when_it_would_quantize`, 무시(ignored) 처리된 실제 체크포인트 테스트 `real_drafter_loads_and_refuses_an_unsupported_scheme`.
- `src/lib/mlxcel-core/src/drafter/dflash/markov_tests.rs`: `a_five_bit_markov_factor_loads`, 7비트 너비 factor가 여전히 거부되는지도 확인.

## 동작 변경

지원되지 않는 `bits`나 `group_size`(0 포함)를 가진 MTP 드래프터 config는 이제 지원되는 값 목록이 담긴 메시지와 함께 `DrafterError::Config`로 로드에 실패한다. 5비트 DFlash Markov factor가 이제 로드된다. 유효한 체크포인트는 이전과 같이 로드된다.

## 검증

- 되돌림 검사: 두 수정을 되돌리면 `quantize_dense_projections_rejects_unsupported_quantization`과 `a_five_bit_markov_factor_loads`가 실패하고, 복원하면 통과한다.
- 실제 드래프터 `mlx-community/Qwen3.8-27B-MTP-bf16`(타깃 `mlx-community/Qwen3.8-27B-4bit`도 다운로드)로 확인: 배포된 드래프터는 변경 없이 로드된다. `bits = 7`, `group_size = 0`인 임시 복사본은 abort 대신 `DrafterError::Config`("quantization bits (7) must be one of 2, 3, 4, 5, 6, 8", "quantization group_size (0) must be one of 32, 64, 128")를 반환한다.
- `qwen3.5-4b-4bit`와 `qwen3.5-4b-dflash`로 `mlxcel-server`를 통해 DFlash를 엔드투엔드 확인(`--draft-kind dflash`, temperature 0): 드래프터가 지연 로드되고 요청이 DFlash 15라운드, 48토큰으로 완료됐다. 처리량 수치는 측정하지 않았다.
- `-D warnings` clippy, fmt, 계약 스크립트가 통과한다.

## 미검증 및 후속

- 엔드투엔드 MTP 추측 생성은 실행하지 않았다. `mlxcel generate --draft-kind mtp`는 기존 게이트로 Metal 전용이라, 이 CUDA 호스트에서는 드래프터를 로드 테스트로만 검증했다.
- 5비트 Markov 경로는 단위 테스트로만 검증했다. 로컬 DFlash 드래프터에 Markov 헤드가 없기 때문이다.
