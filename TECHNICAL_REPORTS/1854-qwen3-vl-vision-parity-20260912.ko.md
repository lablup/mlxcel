# 기술 보고서: PR #1854 - Qwen3-VL 비전 파리티 국소화

**작성일**: 2026-09-12
**상태**: 완료
**언어**: Rust, Python, Markdown
**위험도**: 중간

## 요약

PR #1854는 #1735의 Cohere Compass 구현에서 텍스트 생성은 토큰 단위로 일치하지만 이미지 조건부 continuation은 일치하지 않는다고 확인한 뒤 #1738에서 시작한 이미지 경로 파리티 조사를 이어간다. `mlx-community/North-Micro-Vision-Instruct-4bit`를 고정된 transformers 오라클과 단계별로 비교할 수 있게 만들고, 학습된 위치 임베딩 보간에서 일찍 정밀도를 잃던 문제를 수정하며, 처음 남는 유의미한 발산 지점을 기록한다.

수정된 위치 임베딩은 float32 오라클과 RMSE `2.84e-07`로 일치하고 vision RoPE는 정확히 같다. 일반적인 float32 반올림을 넘어서는 첫 잔차는 block 0 MLP에서 RMSE `0.00228`로 나타나며 post-merger RMSE `0.00810`까지 누적된다. 따라서 이 PR은 이미지 조건부 토큰 정확 일치를 주장하지 않는다.

## 1. 문제 정의

PR #1735는 Cohere Compass 불일치를 공유 27블록 Qwen3-VL 비전 타워로 좁혔다. TF32, 두 가지 비전 activation 후보, DeepStack 순서와 주입 깊이, 이미지 processor 경계, 양자화는 이미 원인에서 배제했다. 남아 있던 근거는 end-to-end 결과뿐이었다. mlxcel은 `The image features...`를 생성하지만 transformers 오라클은 `The image displays...`를 생성했다.

이 근거만으로는 최초의 유의미한 오차가 patch embedding, 학습 위치 임베딩, vision RoPE, attention, MLP, DeepStack, merger 중 어디에서 생겼는지 알 수 없었다. 기존 production forward 경로에는 중간 텐서를 안전하게 추출하는 방법도 없었고 synthetic 비교만으로는 공개 체크포인트의 동작을 증명할 수 없었다.

따라서 조사에는 다음 세 속성이 필요했다.

1. 구현과 같은 processor 및 weight를 사용하는 실제 체크포인트 오라클.
2. 최종 결과뿐 아니라 최초 발산을 찾을 수 있을 만큼 세분된 단계 경계.
3. production 요청 경로 overhead, decoder 변경, video 범위 확장, Apple 하드웨어 세대가 불일치의 원인이라는 가정이 없을 것.

이 국소화가 없으면 이후 수정이 엉뚱한 하위 시스템을 바꾸거나, 의미적 오류를 tolerance 확대로 덮거나, 이미 반증된 가설을 반복할 수 있었다.

## 2. 기술적 선택과 그 이유

### 2.1 두 번째 구현을 만들지 않고 기존 타워를 관찰한다

테스트 전용 `Qwen3VLVisionStage` observer는 입력 patch, patch embedding, 학습 위치 임베딩, 위치 임베딩 추가 후 hidden state, vision RoPE, 모든 블록의 attention/attention 이후/MLP/output, DeepStack 텐서 세 개, merger 전후를 보고한다. ignored dump 테스트는 관찰한 각 텐서를 little-endian float32와 JSON manifest로 저장한다.

observer는 `#[cfg(any(test, feature = "test-utils"))]`로 제한된다. production은 기존 forward 메서드를 계속 호출하므로 일반 추론에는 callback dispatch, tensor copy, file I/O, feature flag가 추가되지 않는다.

### 2.2 보간 연산 구간에서만 float32를 보존한다

기존 경로는 bilinear weight를 float32로 만든 직후 학습 테이블의 bf16/f16 dtype으로 cast했다. 따라서 위치 결과가 hidden state에 도달하기 전에 weight와 multiply/add 모두 정밀도를 잃었다.

```
이전: position table dtype + downcast weight -> 저정밀도 보간 -> residual add
이후: gather한 row를 f32로 cast + f32 weight -> f32 보간 -> hidden dtype으로 한 번 cast -> residual add
```

수정은 gather한 네 embedding corner를 float32로 올리고 weighted sum을 float32에서 수행한 뒤, 완성된 위치 임베딩을 더하기 직전에 patch hidden-state dtype으로 되돌린다. 이 경계는 transformers의 연산과 일치하면서 27개 비전 블록 전체를 float32로 승격하지 않는다.

### 2.3 고정된 오라클의 보간 convention을 따른다

요청된 `torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)` 비교를 수행했지만, 고정한 transformers Cohere Compass/Qwen3-VL 구현은 `bilinear`, `align_corners=True`를 사용한다. 회귀 테스트는 true-corners 결과를 고정하고 half-pixel false-corners fixture가 `2.0` 넘게 다름을 별도로 증명한다. 이후 refactor가 잘못된 좌표 변환을 조용히 채택하지 못하게 한다.

### 2.4 오라클 provenance와 dump 입력을 fail-closed로 만든다

헬퍼는 무관한 변환 weight를 비교하지 않고 #1735의 float32-copy 방식을 재사용한다. 로컬 체크포인트와 `AutoProcessor`를 로드하고 transformers 비전 타워를 재구성해 Rust manifest와 같은 이미지 및 grid를 비교한다.

리뷰에서 최초 헬퍼가 고정 revision을 출력하기만 하고 설치된 package의 출처는 증명하지 않는다는 점을 발견했다. 최종 헬퍼는 설치된 distribution의 `direct_url.json`을 읽고 transformers commit `df04b012229d50d2b6dfba32c61c3057c3a40ea1`을 강제한다.

보안 리뷰에서는 model loading을 `trust_remote_code=False`인 로컬 파일로 제한하고, symlink가 아닌 regular file만 허용하며, stage path가 dump directory 아래에 남는지 확인하고, 양수 shape와 float32 dtype을 검증하며, 정확한 byte length를 확인하고, stage 하나를 최대 1 GiB로 제한했다. 이 검사는 out-of-band 진단 도구가 임의 파일 읽기나 메모리 고갈 경로가 되지 않게 한다.

## 3. 국소화 결과

비교에는 `tests/fixtures/test_image_shapes.png`, grid `[1, 28, 28]`, `mlx-community/North-Micro-Vision-Instruct-4bit`의 float32 copy, 위 commit으로 검증한 transformers `5.18.0.dev0`, torch `2.14.0`을 사용했다.

| 단계 | 최대 절대 오차 | 평균 절대 오차 | RMSE | 해석 |
| --- | ---: | ---: | ---: | --- |
| Rust T,C를 오라클 C,T view로 바꾼 processor 입력 | `2.98e-08` | `9.84e-10` | `5.42e-09` | layout 의미가 동일함 |
| 변환하지 않은 raw T,C 비교 | `1.49` | `0.179` | `0.464` | 예상된 layout 불일치이며 patch 버그가 아님 |
| Patch embedding | `1.14e-05` | `1.25e-07` | `2.04e-07` | float32 반올림 규모 |
| 학습 위치 임베딩 | `5.72e-05` | `3.73e-08` | `2.84e-07` | 수정된 보간 정밀도 경계 |
| Vision RoPE | `0` | `0` | `0` | 정확히 일치 |
| Block 0 attention | `1.29e-05` | `2.37e-07` | `4.98e-07` | float32 반올림 규모 |
| Block 0 MLP | `0.0311` | `0.00143` | `0.00228` | 처음 남는 유의미한 발산 |
| Block 8 이후 DeepStack | `0.0122` | `0.00148` | `0.00188` | 타워를 거치며 누적된 잔차 |
| Block 16 이후 DeepStack | `0.0170` | `0.00154` | `0.00206` | 타워를 거치며 누적된 잔차 |
| Block 24 이후 DeepStack | `0.0243` | `0.00151` | `0.00209` | 타워를 거치며 누적된 잔차 |
| Post-merger | `0.0961` | `0.00602` | `0.00810` | 최종 측정 비전 잔차 |

입력 layout 비교는 잘못된 patch-embedding 수정을 막았다. Rust는 raw patch를 T,C 순서로 저장하고 transformers는 C,T view를 노출한다. 대응하는 reshape와 transpose를 적용하면 입력 차이가 RMSE `0.464`에서 `5.42e-09`로 줄어든다. 따라서 기존 MLX kernel weight 배열을 유지한다.

보간 수정 후에도 실제 greedy prompt는 정확히 일치하지 않는다. mlxcel은 `The image features...`로 시작하고 기록된 오라클은 `The image displays...`로 시작한다. 이 보고서와 supported-model 문서는 block 0 MLP를 다음 국소화 경계로 취급하며, 아직 증명된 root cause라고 부르지 않는다.

## 4. 구현 요약

| 파일 | 책임 |
| --- | --- |
| `src/vision/encoders/qwen3_vl.rs` | Float32 보간 경계와 테스트 전용 stage observer |
| `src/vision/encoders/qwen3_vl_stage_observer_tests.rs` | 정밀도, 보간 convention, patch 순서, dtype, stage coverage 회귀 테스트 |
| `src/vision/encoders/qwen3_vl_stage_dump_tests.rs` | Manifest를 포함한 ignored 실제 체크포인트 float32 stage dump |
| `scripts/tools/qwen3_vl_stage_oracle_compare.py` | 고정 transformers 비교, float32 checkpoint copy, provenance 및 입력 검증 |
| `docs/supported-models.md` | 재현 가능한 측정값과 token non-exactness 명시 |

체크포인트 텐서, 생성된 dump, 모델 binary, decoder 변경, video 지원은 커밋하지 않는다.

## 5. 리뷰 지적사항

| 지적 | 심각도 | 해결 |
| --- | --- | --- |
| 설치된 transformers provenance를 주장했지만 검증하지 않음 | 높음 | `55ccd5f7`에서 `direct_url.json` commit metadata를 검증하도록 수정 |
| 로컬 manifest/checkpoint 입력이 안전하지 않은 path를 따르거나 무제한 read를 요청할 수 있음 | 중간 | `07ac1462`에서 local-only loading, regular-file/path 검사, 정확한 size, stage당 1 GiB 제한 추가 |

남은 Critical 또는 High 지적은 없다. production 성능 변화는 학습 위치 임베딩 보간 자체의 float32 연산으로 제한된다. 완성된 embedding은 27블록 타워 전에 다시 cast하므로 hidden state의 지속적 승격을 피한다.

## 6. 검증

다음 로컬 gate가 통과했다.

- `cargo test --profile test-fast --features metal,accelerate --lib vision::encoders::qwen3_vl::` — 8개 통과, 1개 ignored.
- ignored `dump_north_micro_vision_stage_tensors` 테스트를 실제 체크포인트에 실행해 오라클 헬퍼가 읽은 manifest를 생성했다.
- `/private/tmp/qwen3vl-oracle-1738/bin/python scripts/tools/qwen3_vl_stage_oracle_compare.py compare --model models/North-Micro-Vision-Instruct-4bit-f32 --rust-dump /private/tmp/qwen3vl-stage-1738-final --model-dtype f32 --input-dtype f32`가 3절의 측정값을 재현하고 오라클 provenance를 검증했다.
- `cargo check --lib --tests --features metal,accelerate`.
- `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`.
- `cargo fmt --all -- --check`.
- Python bytecode compile, cross-repository reference 검사, `git diff --check origin/main...HEAD`.

PR CI의 cargo-clippy, cargo-deny, cargo-fmt, OpenXLA feature compile, crate version, cross-repository reference, kernel dtype key, license header, WebUI contract, llama compatibility, CLA 검사가 통과했다. 플랫폼상 무관한 job은 skip됐다.

## 7. 변경 요약

이 보고서를 추가하기 전 구현 diff는 5개 파일, 1,042줄 추가, 5줄 삭제다.

| 커밋 | 목적 |
| --- | --- |
| `5d5c9ef9` | Qwen3-VL stage를 국소화하고 position interpolation 정밀도 수정 |
| `55ccd5f7` | transformers 오라클 provenance 검증 |
| `07ac1462` | 로컬 오라클 입력 강화 |

이슈 #1738은 `Closes #1738`로 연결되어 있다. PR은 wave runner가 머지하도록 의도적으로 open 상태로 남긴다.

## 8. 후속 조치와 학습 포인트

다음 파리티 조사는 block 0 MLP 내부에서 시작해 norm output, 첫 projection, tanh-GELU, gate/product, 두 번째 projection을 비교한 뒤 이후 블록의 residual을 따라가야 한다. 첫 MLP 잔차가 설명되기 전까지 decoder 및 video 변경은 범위 밖이다.

이 작업에서 재사용할 수 있는 교훈은 두 가지다.

- raw 저장 순서가 아니라 tensor 의미를 비교해야 한다. T,C/C,T 입력 결과를 잘못 해석했다면 그럴듯하지만 틀린 patch weight 변경이 들어갔을 것이다.
- 오라클 고정과 검증을 측정의 일부로 취급해야 한다. version string이나 출력된 상수는 provenance가 아니다. 또한 #1769는 공유 software pin이 byte-identical 동작을 만들 수 있을 때 설명되지 않은 실패를 M1/M5 하드웨어 세대 차이로 귀속하면 안 되는 이유를 이미 보여줬다.

관련 배경: [`1735-cohere-compass-vlm-20260910.ko.md`](1735-cohere-compass-vlm-20260910.ko.md), 특히 오라클 구성과 이미지 경로 조사 결과.
