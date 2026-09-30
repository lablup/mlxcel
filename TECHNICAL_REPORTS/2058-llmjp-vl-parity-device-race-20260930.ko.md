# 기술 보고서: 이슈 #2058 - llmjp_vl conv 레이아웃 동등성 테스트의 기본 디바이스 경합

## 요약

`vision::llmjp_vl::tests::either_patch_embedding_conv_layout_produces_the_same_features`는 M1 Ultra의 전체 `cargo test --release -p mlxcel --lib` 15회 중 2회 실패했고, 매번 `max diff 0.000000018626451`이었다. 단독 실행은 통과했다. 이 테스트는 잠금을 잡지 않아서, MLX의 프로세스 전역 기본 디바이스를 CPU로 옮기는 다른 테스트(`lock_default_device()` 아래의 `DefaultDeviceGuard::cpu()`)와 겹치면 한 타워는 CPU, 다른 타워는 Metal에서 실행될 수 있었다. CPU와 Metal은 마지막 비트에서 차이가 나므로 정확 비교 `== 0.0`이 깨졌다.

## 원인 증명

수정을 적용한 상태에서 두 forward 사이에 `let _cpu = DefaultDeviceGuard::cpu();` 프로브를 넣으면 테스트가 결정적으로 실패했고, 값은 soak 실패와 같은 `max diff 0.000000018626451`이었다. 프로브는 커밋 전에 제거했다.

가중치 레이아웃은 원인이 아니다. 두 레이아웃 모두 같은 strided view로 conv에 도달한다(MLX `copy`는 버퍼를 공유하고, Metal conv는 dispatch 전에 가중치를 row-contiguous로 만든다).

## 변경

- `src/vision/llmjp_vl_tests.rs`: 테스트가 `mlx_test_guard()`를 가장 먼저 잡는다. 이 가드는 기본 디바이스 잠금을 잡고 기본 디바이스가 GPU인지 확인한다. 단언은 `max_diff == 0.0`을 유지하며 허용 오차는 추가하지 않았다.
- `src/vision/encoders/siglip.rs`: 새 테스트 `patch_embed_both_conv_layouts_yield_identical_weight`가 같은 값의 HF 레이아웃 맵과 MLX 레이아웃 맵으로 `VisionEmbeddings::from_weights`를 만들고, 두 가중치의 shape가 `[O, kH, kW, I]`이며 원소 단위로 정확히 같음을 단언한다. `from_weights`의 transpose를 `[0, 2, 3, 1]`에서 `[0, 3, 2, 1]`로 바꾸면 실패한다.

## 검증

- 전체 `cargo test --release -p mlxcel --lib` 20회 연속 실행: 20회 모두 통과(회당 8726개), 실패 없음.
- `cargo fmt --all -- --check`, `cargo clippy --release -p mlxcel --lib --tests -- -D warnings`, `cli_help_consistency`, `dead_doc_pointers`, `llama_compat_manifest` 계약 테스트 통과.

## 변경하지 않은 것

다른 정확 비교 vision 테스트는 조사만 하고 손대지 않았다(이슈 범위 밖). 자세한 내용은 PR 설명을 참고한다.
